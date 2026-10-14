#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint64_t frames;
    uint64_t dropped;
    uint64_t first_wall_ms;
    uint64_t last_wall_ms;
} V420RecordingStats;

static void copy_error(char *destination, size_t capacity, NSString *message) {
    if (capacity > 0) {
        snprintf(destination, capacity, "%s", message.UTF8String ?: "Unknown camera error");
    }
}

@interface V420ReferenceRecorder : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate> {
@public
    V420RecordingStats stats;
    AVAssetWriter *_writer;
    AVAssetWriterInput *_writerInput;
    AVAssetWriterInputPixelBufferAdaptor *_adaptor;
    NSURL *_videoURL;
    FILE *_timestamps;
    NSString *_failure;
    AVCaptureSession *_session;
    AVCaptureVideoDataOutput *_videoOutput;
    dispatch_queue_t _captureQueue;
    NSCondition *_frameReady;
    NSData *_latestRGB;
    uint32_t _latestWidth;
    uint32_t _latestHeight;
    uint64_t _latestWallMs;
    uint64_t _sequence;
    uint64_t _consumedSequence;
    int64_t _lastVideoTick;
    uint32_t _writerWidth;
    uint32_t _writerHeight;
    BOOL _stopped;
}
@property(nonatomic, strong) NSString *failure;
- (BOOL)appendRGB:(const uint8_t *)rgb width:(uint32_t)width height:(uint32_t)height
           wallMs:(uint64_t)wallMs error:(char *)error capacity:(size_t)capacity;
- (BOOL)waitForFirstFrame:(NSTimeInterval)timeout;
- (void)stopCapture;
- (BOOL)finish:(char *)error capacity:(size_t)capacity;
@end

@implementation V420ReferenceRecorder
- (instancetype)init {
    if ((self = [super init])) {
        _captureQueue = dispatch_queue_create("420vision.reference.capture", DISPATCH_QUEUE_SERIAL);
        _frameReady = [NSCondition new];
    }
    return self;
}

- (void)dealloc {
    if (_timestamps) fclose(_timestamps);
}

- (BOOL)startWriterWithSampleWidth:(uint32_t)width height:(uint32_t)height
                              error:(char *)error capacity:(size_t)capacity {
    NSError *writerError = nil;
    _writer = [[AVAssetWriter alloc] initWithURL:_videoURL
                                        fileType:AVFileTypeQuickTimeMovie
                                           error:&writerError];
    if (!_writer) {
        self.failure = writerError.localizedDescription ?: @"Cannot create reference video writer";
        copy_error(error, capacity, self.failure);
        return NO;
    }

    NSDictionary *settings = @{
        AVVideoCodecKey: AVVideoCodecTypeH264,
        AVVideoWidthKey: @(width),
        AVVideoHeightKey: @(height),
        AVVideoCompressionPropertiesKey: @{ AVVideoAverageBitRateKey: @4000000 }
    };
    _writerInput = [[AVAssetWriterInput alloc] initWithMediaType:AVMediaTypeVideo
                                                 outputSettings:settings];
    _writerInput.expectsMediaDataInRealTime = YES;
    NSDictionary *pixelAttributes = @{
        (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (NSString *)kCVPixelBufferWidthKey: @(width),
        (NSString *)kCVPixelBufferHeightKey: @(height),
        (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{}
    };
    _adaptor = [AVAssetWriterInputPixelBufferAdaptor
        assetWriterInputPixelBufferAdaptorWithAssetWriterInput:_writerInput
                                   sourcePixelBufferAttributes:pixelAttributes];
    _writerWidth = width;
    _writerHeight = height;
    if (![_writer canAddInput:_writerInput]) {
        self.failure = @"Cannot add reference video writer input";
        copy_error(error, capacity, self.failure);
        return NO;
    }
    [_writer addInput:_writerInput];
    if (![_writer startWriting]) {
        self.failure = _writer.error.localizedDescription ?: @"Cannot start reference video writer";
        copy_error(error, capacity, self.failure);
        return NO;
    }
    [_writer startSessionAtSourceTime:kCMTimeZero];
    return YES;
}

- (BOOL)appendRGB:(const uint8_t *)rgb width:(uint32_t)width height:(uint32_t)height
           wallMs:(uint64_t)wallMs error:(char *)error capacity:(size_t)capacity {
    if (self.failure) {
        copy_error(error, capacity, self.failure);
        return NO;
    }
    if (!rgb || width == 0 || height == 0 || width > 1920 || height > 1080) {
        self.failure = @"Invalid reference frame dimensions or pixels";
        copy_error(error, capacity, self.failure);
        return NO;
    }
    if (!_writer && ![self startWriterWithSampleWidth:width height:height error:error capacity:capacity]) {
        return NO;
    }
    if (width != _writerWidth || height != _writerHeight) {
        self.failure = @"Reference camera resolution changed during recording";
        copy_error(error, capacity, self.failure);
        return NO;
    }
    if (!_adaptor.pixelBufferPool) {
        self.failure = @"Reference video pixel buffer pool is unavailable";
        copy_error(error, capacity, self.failure);
        return NO;
    }
    if (!_writerInput.readyForMoreMediaData) {
        stats.dropped++;
        return YES;
    }

    CVPixelBufferRef pixels = NULL;
    CVReturn allocation = CVPixelBufferPoolCreatePixelBuffer(
        kCFAllocatorDefault, _adaptor.pixelBufferPool, &pixels);
    if (allocation != kCVReturnSuccess || !pixels) {
        self.failure = [NSString stringWithFormat:@"Cannot allocate reference frame (%d)", allocation];
        copy_error(error, capacity, self.failure);
        return NO;
    }
    if (CVPixelBufferLockBaseAddress(pixels, 0) != kCVReturnSuccess) {
        CFRelease(pixels);
        self.failure = @"Cannot lock reference frame buffer";
        copy_error(error, capacity, self.failure);
        return NO;
    }
    uint8_t *destination = CVPixelBufferGetBaseAddress(pixels);
    size_t stride = CVPixelBufferGetBytesPerRow(pixels);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const uint8_t *source = rgb + (y * width + x) * 3;
            uint8_t *pixel = destination + y * stride + x * 4;
            pixel[0] = source[2];
            pixel[1] = source[1];
            pixel[2] = source[0];
            pixel[3] = 255;
        }
    }
    CVPixelBufferUnlockBaseAddress(pixels, 0);

    if (stats.frames == 0) stats.first_wall_ms = wallMs;
    uint64_t elapsedMs = wallMs >= stats.first_wall_ms ? wallMs - stats.first_wall_ms : 0;
    int64_t videoTick = (int64_t)llround((double)elapsedMs * 600.0 / 1000.0);
    if (stats.frames > 0 && videoTick <= _lastVideoTick) videoTick = _lastVideoTick + 1;
    BOOL appended = [_adaptor appendPixelBuffer:pixels
                           withPresentationTime:CMTimeMake(videoTick, 600)];
    CFRelease(pixels);
    if (!appended) {
        self.failure = _writer.error.localizedDescription ?: @"Cannot append reference video frame";
        copy_error(error, capacity, self.failure);
        return NO;
    }

    double videoSeconds = (double)videoTick / 600.0;
    if (fprintf(_timestamps, "%llu,%.6f,%llu\n", (unsigned long long)stats.frames,
                videoSeconds, (unsigned long long)wallMs) < 0) {
        self.failure = @"Cannot write reference frame timestamps";
        copy_error(error, capacity, self.failure);
        return NO;
    }
    _lastVideoTick = videoTick;
    stats.last_wall_ms = wallMs;
    stats.frames++;
    return YES;
}

- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sample
       fromConnection:(AVCaptureConnection *)connection {
    CVPixelBufferRef pixels = CMSampleBufferGetImageBuffer(sample);
    if (!pixels || CVPixelBufferGetPixelFormatType(pixels) != kCVPixelFormatType_32BGRA) return;
    size_t width = CVPixelBufferGetWidth(pixels);
    size_t height = CVPixelBufferGetHeight(pixels);
    if (width == 0 || height == 0 || width > 1920 || height > 1080) return;
    if (CVPixelBufferLockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) return;
    const uint8_t *source = CVPixelBufferGetBaseAddress(pixels);
    size_t stride = CVPixelBufferGetBytesPerRow(pixels);
    NSMutableData *rgb = [NSMutableData dataWithLength:width * height * 3];
    uint8_t *destination = rgb.mutableBytes;
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const uint8_t *pixel = source + y * stride + x * 4;
            uint8_t *rgbPixel = destination + (y * width + x) * 3;
            rgbPixel[0] = pixel[2];
            rgbPixel[1] = pixel[1];
            rgbPixel[2] = pixel[0];
        }
    }
    CVPixelBufferUnlockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly);

    CMTime pts = CMSampleBufferGetPresentationTimeStamp(sample);
    CMTime hostNow = CMClockGetTime(CMClockGetHostTimeClock());
    double age = CMTimeGetSeconds(CMTimeSubtract(hostNow, pts));
    uint64_t wallMs = (uint64_t)llround((NSDate.date.timeIntervalSince1970 - age) * 1000.0);
    char error[1024] = {0};
    if (![self appendRGB:rgb.bytes width:(uint32_t)width height:(uint32_t)height
                  wallMs:wallMs error:error capacity:sizeof(error)]) {
        return;
    }

    [_frameReady lock];
    _latestRGB = [rgb copy];
    _latestWidth = (uint32_t)width;
    _latestHeight = (uint32_t)height;
    _latestWallMs = wallMs;
    _sequence++;
    [_frameReady signal];
    [_frameReady unlock];
}

- (BOOL)waitForFirstFrame:(NSTimeInterval)timeout {
    [_frameReady lock];
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
    while (_sequence == 0 && !_stopped) {
        if (![_frameReady waitUntilDate:deadline]) break;
    }
    BOOL received = _sequence > 0;
    [_frameReady unlock];
    return received;
}

- (void)stopCapture {
    if (_session) {
        [_session stopRunning];
        [_videoOutput setSampleBufferDelegate:nil queue:NULL];
        dispatch_sync(_captureQueue, ^{});
    }
    [_frameReady lock];
    _stopped = YES;
    [_frameReady broadcast];
    [_frameReady unlock];
}

- (BOOL)finish:(char *)error capacity:(size_t)capacity {
    [self stopCapture];
    if (_writer.status == AVAssetWriterStatusWriting) {
        [_writerInput markAsFinished];
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [_writer finishWritingWithCompletionHandler:^{ dispatch_semaphore_signal(done); }];
        if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) != 0) {
            [_writer cancelWriting];
            self.failure = @"Timed out finalizing reference video";
        }
    }
    if (_timestamps) {
        if (fclose(_timestamps) != 0) self.failure = @"Cannot flush reference frame timestamps";
        _timestamps = NULL;
    }
    if (stats.frames == 0) self.failure = @"No video frames recorded";
    if (self.failure || _writer.status != AVAssetWriterStatusCompleted) {
        copy_error(error, capacity,
                   self.failure ?: _writer.error.localizedDescription ?: @"Reference video did not finalize");
        return NO;
    }
    return YES;
}
@end

static V420ReferenceRecorder *create_recorder(const char *video, const char *timestamps,
        char *error, size_t errorCapacity) {
    V420ReferenceRecorder *recorder = [V420ReferenceRecorder new];
    recorder->_videoURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:video]];
    recorder->_timestamps = fopen(timestamps, "wx");
    if (!recorder->_timestamps) {
        copy_error(error, errorCapacity, @"Cannot create reference timestamp file");
        return nil;
    }
    if (fprintf(recorder->_timestamps, "frame_index,video_seconds,capture_wall_ms\n") < 0) {
        copy_error(error, errorCapacity, @"Cannot write reference timestamp header");
        return nil;
    }
    return recorder;
}

void *v420_reference_capture_start(const char *deviceID, const char *video, const char *timestamps,
        char *error, size_t errorCapacity) {
    @autoreleasepool {
        V420ReferenceRecorder *camera = create_recorder(video, timestamps, error, errorCapacity);
        if (!camera) return NULL;

        AVAuthorizationStatus auth = [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
        if (auth == AVAuthorizationStatusNotDetermined) {
            dispatch_semaphore_t authorized = dispatch_semaphore_create(0);
            __block BOOL granted = NO;
            [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo
                                    completionHandler:^(BOOL allowed) {
                granted = allowed;
                dispatch_semaphore_signal(authorized);
            }];
            if (dispatch_semaphore_wait(authorized, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC)) != 0) {
                copy_error(error, errorCapacity, @"Timed out waiting for macOS camera permission");
                return NULL;
            }
            auth = granted ? AVAuthorizationStatusAuthorized : AVAuthorizationStatusDenied;
        }
        if (auth != AVAuthorizationStatusAuthorized) {
            copy_error(error, errorCapacity,
                       [NSString stringWithFormat:@"macOS camera permission is not authorized (status %ld)", (long)auth]);
            return NULL;
        }

        AVCaptureDevice *device = [AVCaptureDevice deviceWithUniqueID:
            [NSString stringWithUTF8String:deviceID]];
        NSError *inputError = nil;
        AVCaptureDeviceInput *input = device
            ? [AVCaptureDeviceInput deviceInputWithDevice:device error:&inputError] : nil;
        if (!input) {
            copy_error(error, errorCapacity, inputError.localizedDescription ?: @"Selected camera is unavailable");
            return NULL;
        }

        camera->_session = [AVCaptureSession new];
        [camera->_session beginConfiguration];
        if ([camera->_session canSetSessionPreset:AVCaptureSessionPreset1280x720]) {
            camera->_session.sessionPreset = AVCaptureSessionPreset1280x720;
        }
        camera->_videoOutput = [AVCaptureVideoDataOutput new];
        camera->_videoOutput.videoSettings = @{
            (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)
        };
        camera->_videoOutput.alwaysDiscardsLateVideoFrames = YES;
        if (![camera->_session canAddInput:input] || ![camera->_session canAddOutput:camera->_videoOutput]) {
            [camera->_session commitConfiguration];
            copy_error(error, errorCapacity, @"Cannot configure reference camera session");
            return NULL;
        }
        [camera->_session addInput:input];
        [camera->_session addOutput:camera->_videoOutput];
        [camera->_videoOutput setSampleBufferDelegate:camera queue:camera->_captureQueue];
        [camera->_session commitConfiguration];
        [camera->_session startRunning];
        if (!camera->_session.running) {
            [camera->_videoOutput setSampleBufferDelegate:nil queue:NULL];
            copy_error(error, errorCapacity, @"Reference camera session did not start");
            return NULL;
        }
        if (![camera waitForFirstFrame:5.0]) {
            [camera stopCapture];
            copy_error(error, errorCapacity,
                       @"Camera opened but delivered no frames within 5 seconds; verify app camera permission and camera ownership");
            return NULL;
        }
        return (__bridge_retained void *)camera;
    }
}

// 1 = latest frame copied, 0 = timed out/stopped, -1 = invalid buffer.
int v420_reference_capture_frame(void *handle, uint8_t *destination, size_t capacity,
        uint32_t *width, uint32_t *height, uint64_t *wallMs, uint32_t timeoutMs) {
    @autoreleasepool {
        V420ReferenceRecorder *camera = (__bridge V420ReferenceRecorder *)handle;
        [camera->_frameReady lock];
        NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:(NSTimeInterval)timeoutMs / 1000.0];
        while (camera->_sequence == camera->_consumedSequence && !camera->_stopped) {
            if (![camera->_frameReady waitUntilDate:deadline]) break;
        }
        if (camera->_stopped || camera->_sequence == camera->_consumedSequence) {
            [camera->_frameReady unlock];
            return 0;
        }
        NSData *rgb = camera->_latestRGB;
        if (!rgb || rgb.length > capacity) {
            [camera->_frameReady unlock];
            return -1;
        }
        memcpy(destination, rgb.bytes, rgb.length);
        *width = camera->_latestWidth;
        *height = camera->_latestHeight;
        *wallMs = camera->_latestWallMs;
        camera->_consumedSequence = camera->_sequence;
        [camera->_frameReady unlock];
        return 1;
    }
}

int v420_reference_writer_stop(void *handle, char *error, size_t capacity,
        V420RecordingStats *stats) {
    @autoreleasepool {
        V420ReferenceRecorder *recorder = (__bridge_transfer V420ReferenceRecorder *)handle;
        BOOL completed = [recorder finish:error capacity:capacity];
        *stats = recorder->stats;
        return completed ? 1 : 0;
    }
}

void *v420_reference_writer_start(const char *video, const char *timestamps,
        char *error, size_t errorCapacity) {
    @autoreleasepool {
        V420ReferenceRecorder *recorder = create_recorder(video, timestamps, error, errorCapacity);
        return recorder ? (__bridge_retained void *)recorder : NULL;
    }
}

int v420_reference_writer_append(void *handle, const uint8_t *rgb, uint32_t width, uint32_t height,
        uint64_t wallMs, char *error, size_t errorCapacity) {
    @autoreleasepool {
        V420ReferenceRecorder *recorder = (__bridge V420ReferenceRecorder *)handle;
        return [recorder appendRGB:rgb width:width height:height wallMs:wallMs
                             error:error capacity:errorCapacity] ? 1 : 0;
    }
}

int64_t v420_reference_test_decoded_frames(const char *video) {
    @autoreleasepool {
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:video]] options:nil];
        __block NSArray<AVAssetTrack *> *tracks = nil;
        dispatch_semaphore_t loaded = dispatch_semaphore_create(0);
        [asset loadTracksWithMediaType:AVMediaTypeVideo completionHandler:
            ^(NSArray<AVAssetTrack *> *value, NSError *loadError) {
                tracks = loadError ? nil : value;
                dispatch_semaphore_signal(loaded);
            }];
        if (dispatch_semaphore_wait(loaded, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0
                || tracks.count != 1) return -1;
        NSError *error = nil;
        AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:&error];
        AVAssetReaderTrackOutput *output = [AVAssetReaderTrackOutput
            assetReaderTrackOutputWithTrack:tracks.firstObject
                             outputSettings:@{
                                 (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)
                             }];
        if (!reader || ![reader canAddOutput:output]) return -1;
        [reader addOutput:output];
        if (![reader startReading]) return -1;
        int64_t count = 0;
        CMSampleBufferRef sample;
        while ((sample = [output copyNextSampleBuffer])) {
            CVPixelBufferRef pixels = CMSampleBufferGetImageBuffer(sample);
            BOOL valid = pixels && CVPixelBufferGetWidth(pixels) == 64
                && CVPixelBufferGetHeight(pixels) == 48;
            CFRelease(sample);
            if (!valid) return -1;
            count++;
        }
        return reader.status == AVAssetReaderStatusCompleted ? count : -1;
    }
}
