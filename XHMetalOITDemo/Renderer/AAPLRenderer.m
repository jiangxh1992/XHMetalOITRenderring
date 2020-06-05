@import simd;
@import ModelIO;
@import MetalKit;

#import "AAPLRenderer.h"
#import "AAPLMesh.h"
#import "AAPLMathUtilities.h"
#import "AAPLShaderTypes.h"

static const NSUInteger AAPLMaxBuffersInFlight = 3;

@implementation AAPLRenderer
{
    CGSize _windowSize;

    NSUInteger _frameNum;
    
    dispatch_semaphore_t _inFlightSemaphore;
    id <MTLDevice> _device;
    id <MTLCommandQueue> _commandQueue;

    // Buffer for uniforms which change per-frame
    id <MTLBuffer> _frameUniformBuffers[AAPLMaxBuffersInFlight];
    
    // We have a fragment shader per rendering method
    id <MTLRenderPipelineState> _pipelineState;
    id <MTLDepthStencilState> _depthState;
    
    // Tile shader used to prepare the imageblock memory
    id <MTLRenderPipelineState> _clearTileState;
    
    // Tile shader used to resolve the imageblock OIT data into the final framebuffer
    id <MTLRenderPipelineState> _resolveState;

    // Metal vertex descriptor specifying how vertices will by laid out  render
    //   for input into our pipeline and how we'll layout our ModelIO vertices
    MTLVertexDescriptor *_mtlVertexDescriptor;

    // Used to determine _uniformBufferStride each frame.
    //   This is the current frame number modulo AAPLMaxBuffersInFlight
    uint8_t _uniformBufferIndex;

    // Projection matrix calculated as a function of view size
    matrix_float4x4 _projectionMatrix;

    // Current rotation of our object in radians
    float _rotation;

    // Array of App-Specific mesh objects in our scene
    NSMutableArray<MTKMesh *> *_meshes;

    // Buffer that holds OIT data if device memory is being used
    id <MTLBuffer> _oitBufferData;
    id <MTLTexture> _colorMap;

}

-(nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)view
{
    self = [super init];
    if(self)
    {
        _device = view.device;
        if(![_device supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily4_v1])
        {
            assert(!"Sample requires GPUFamily4_v1 (introduced with A11)");
            return nil;
        }
        _inFlightSemaphore = dispatch_semaphore_create(AAPLMaxBuffersInFlight);
        [self loadMetalWithMetalKitView:view];
        [self loadAssets];
    }
    return self;
}

/// Create and load our basic Metal state objects
- (void)loadMetalWithMetalKitView:(nonnull MTKView *)view
{
    NSError *error;
    // Load all the shader files with a metal file extension in the project
    id <MTLLibrary> defaultLibrary = [_device newDefaultLibrary];
    // Create and allocate uniform buffer objects.
    for(NSUInteger i = 0; i < AAPLMaxBuffersInFlight; i++)
    {
        // Indicate shared storage so that both the CPU and GPU can access the buffer
        const MTLResourceOptions storageMode = MTLResourceStorageModeShared;
        _frameUniformBuffers[i] = [_device newBufferWithLength:sizeof(AAPLFrameUniforms)
                                                       options:storageMode];
        _frameUniformBuffers[i].label = [NSString stringWithFormat:@"FrameUniformBuffer%lu", i];
    }
    // Function constants for the functions
    MTLFunctionConstantValues *constantValues = [MTLFunctionConstantValues new];
    // Load the various fragment functions into the library
    id <MTLFunction> transparencyMethodFragmentFunction =
    [defaultLibrary newFunctionWithName:@"OITFragmentFunction" constantValues:constantValues error:nil];
    // Load the vertex function into the library
    id <MTLFunction> vertexFunction = [defaultLibrary newFunctionWithName:@"vertexTransform"];
    id <MTLFunction> resolveFunction =
    [defaultLibrary newFunctionWithName:@"OITResolve" constantValues:constantValues error:nil];
    id <MTLFunction> clearFunction =
        [defaultLibrary newFunctionWithName:@"OITClear" constantValues:constantValues error:nil];

    // Create a vertex descriptor for our Metal pipeline. Specifies the layout of vertices the
    //   pipeline should expect.  The layout below keeps attributes used to calculate vertex shader
    //   output position separate (world position, skinning, tweening weights) separate from other
    //   attributes (texture coordinates, normals).  This generally maximizes pipeline efficiency
    _mtlVertexDescriptor = [[MTLVertexDescriptor alloc] init];

    _mtlVertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    _mtlVertexDescriptor.attributes[0].offset = 0;
    _mtlVertexDescriptor.attributes[0].bufferIndex = 0;

    _mtlVertexDescriptor.attributes[1].format = MTLVertexFormatFloat2;
    _mtlVertexDescriptor.attributes[1].offset = 12;
    _mtlVertexDescriptor.attributes[1].bufferIndex = 0;
    
    _mtlVertexDescriptor.attributes[2].format = MTLVertexFormatHalf4;
    _mtlVertexDescriptor.attributes[2].offset = 20;
    _mtlVertexDescriptor.attributes[2].bufferIndex = 0;

    _mtlVertexDescriptor.layouts[0].stride = 44;
    _mtlVertexDescriptor.layouts[0].stepRate = 1;
    _mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;

    view.depthStencilPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
    view.sampleCount = 1;
    
    // Create a reusable pipeline state
    MTLRenderPipelineDescriptor *pipelineStateDescriptor =
        [[MTLRenderPipelineDescriptor alloc] init];
    pipelineStateDescriptor.vertexDescriptor = _mtlVertexDescriptor;
    pipelineStateDescriptor.vertexFunction = vertexFunction;
    pipelineStateDescriptor.sampleCount = view.sampleCount;
    pipelineStateDescriptor.depthAttachmentPixelFormat = view.depthStencilPixelFormat;
    pipelineStateDescriptor.stencilAttachmentPixelFormat = view.depthStencilPixelFormat;
    pipelineStateDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;

    // Create the pipeline state

    // We will not actually write colors with our render pass when using or OIT methods
    //    Instead, our tile shaders will accomplish these writes
    pipelineStateDescriptor.colorAttachments[0].blendingEnabled = NO;
    pipelineStateDescriptor.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
    pipelineStateDescriptor.fragmentFunction = transparencyMethodFragmentFunction;
    _pipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineStateDescriptor error:&error];
    if (!_pipelineState)
    {
            NSLog(@"Failed to create pipeline state, error %@", error);
    }

    // Create the various tile states for setting up and resolving imageblock memory
    // because of the usage of tile render pipeline descriptors
    for (NSUInteger i = 0; i < 1; ++i)
    {
        MTLTileRenderPipelineDescriptor *tileDesc = [MTLTileRenderPipelineDescriptor new];
        tileDesc.tileFunction = resolveFunction;
        tileDesc.colorAttachments[0].pixelFormat = view.colorPixelFormat;
        tileDesc.threadgroupSizeMatchesTileSize = YES;
        _resolveState = [_device newRenderPipelineStateWithTileDescriptor:tileDesc
                                                                      options:0
                                                                   reflection:nil
                                                                        error:&error];
        if (!_resolveState)
        {
            NSLog(@"Failed to create tile pipeline state, error %@", error);
        }

        tileDesc.tileFunction = clearFunction;
        _clearTileState = [_device newRenderPipelineStateWithTileDescriptor:tileDesc
                                                                        options:0
                                                                     reflection:nil
                                                                          error:&error];
        if (!_clearTileState)
        {
            NSLog(@"Failed to create tile pipeline state, error %@", error);
        }
    }

    MTLDepthStencilDescriptor *depthStateDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStateDesc.depthCompareFunction = MTLCompareFunctionAlways;
    depthStateDesc.depthWriteEnabled = NO;
    _depthState = [_device newDepthStencilStateWithDescriptor:depthStateDesc];
    // Create the command queue
    _commandQueue = [_device newCommandQueue];
    _frameNum = 0;
}

/// Create and load our assets into Metal objects including meshes and textures
- (void)loadAssets
{
    _meshes = [[NSMutableArray alloc] init];
	NSError *error;
    MDLVertexDescriptor *mdlVertexDescriptor =
    MTKModelIOVertexDescriptorFromMetal(_mtlVertexDescriptor);
    mdlVertexDescriptor.attributes[0].name  = MDLVertexAttributePosition;
    mdlVertexDescriptor.attributes[1].name  = MDLVertexAttributeTextureCoordinate;
    mdlVertexDescriptor.attributes[2].name  = MDLVertexAttributeNormal;

    MTKMeshBufferAllocator *metalAllocator = [[MTKMeshBufferAllocator alloc]
                                              initWithDevice: _device];
    MDLMesh *cubemesh = [MDLMesh newBoxWithDimensions:(vector_float3){4, 4, 4}
                                            segments:(vector_uint3){2, 2, 2}
                                        geometryType:MDLGeometryTypeTriangles
                                       inwardNormals:NO
                                           allocator:metalAllocator];
    MDLMesh *cubemesh2 = [MDLMesh newBoxWithDimensions:(vector_float3){4.5, 4.5, 4.5}
         segments:(vector_uint3){1.5, 1.5, 1.5}
     geometryType:MDLGeometryTypeTriangles
    inwardNormals:NO
        allocator:metalAllocator];
    
    MDLMesh *capsulemesh = [MDLMesh newCapsuleWithHeight:5 radii:(vector_float2){1, 1} radialSegments:10 verticalSegments:10 hemisphereSegments:20 geometryType:MDLGeometryTypeTriangles inwardNormals:NO allocator:metalAllocator];
    MDLMesh *capsulemesh2 = [MDLMesh newCapsuleWithHeight:6 radii:(vector_float2){1.5, 1.5} radialSegments:10 verticalSegments:10 hemisphereSegments:20 geometryType:MDLGeometryTypeTriangles inwardNormals:NO allocator:metalAllocator];
    MDLMesh *ellipmesh = [MDLMesh newEllipsoidWithRadii:(vector_float3){2,2,2} radialSegments:50 verticalSegments:50 geometryType:MDLGeometryTypeTriangles inwardNormals:NO hemisphere:NO allocator:metalAllocator];

    cubemesh.vertexDescriptor = mdlVertexDescriptor;
    capsulemesh.vertexDescriptor = mdlVertexDescriptor;
    cubemesh2.vertexDescriptor = mdlVertexDescriptor;
    capsulemesh2.vertexDescriptor = mdlVertexDescriptor;
    ellipmesh.vertexDescriptor = mdlVertexDescriptor;
    MTKMesh *cube = [[MTKMesh alloc] initWithMesh:cubemesh device:_device error:&error];
    MTKMesh *capsule = [[MTKMesh alloc] initWithMesh:capsulemesh device:_device error:&error];
    MTKMesh *cube2 = [[MTKMesh alloc] initWithMesh:cubemesh2 device:_device error:&error];
    MTKMesh *capsule2 = [[MTKMesh alloc] initWithMesh:capsulemesh2 device:_device error:&error];
    MTKMesh *ellip = [[MTKMesh alloc] initWithMesh:ellipmesh device:_device error:&error];
    
    [_meshes addObject:cube];
    [_meshes addObject:capsule];
    //[_meshes addObject:cube2];
    //[_meshes addObject:capsule2];
    //[_meshes addObject:ellip];
    
    MTKTextureLoader* textureLoader = [[MTKTextureLoader alloc] initWithDevice:_device];

    NSDictionary *textureLoaderOptions =
    @{
      MTKTextureLoaderOptionTextureUsage       : @(MTLTextureUsageShaderRead),
      MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModePrivate)
      };
    _colorMap = [textureLoader newTextureWithName:@"FoliageBaseColorMap"
                                      scaleFactor:1.0
                                           bundle:nil
                                          options:textureLoaderOptions
                                            error:&error];
}

/// Update the state of our "Game" for the current frame
- (void)updateGameState
{
    // Update any game state (including updating dynamically changing Metal buffer)
    _uniformBufferIndex = (_uniformBufferIndex + 1) % AAPLMaxBuffersInFlight;
    AAPLFrameUniforms *uniforms =
        (AAPLFrameUniforms *)_frameUniformBuffers[_uniformBufferIndex].contents;
    uniforms->projectionMatrix = _projectionMatrix;
    uniforms->viewMatrix = matrix4x4_translation(0.0, 0, 10.0);
    vector_float3 rotationAxis = {0, 1, 0};
    matrix_float4x4 modelMatrix = matrix4x4_rotation(_rotation, rotationAxis);
    //matrix_float4x4 translation = matrix4x4_translation(0.0, -200, 0);
    //modelMatrix = matrix_multiply(modelMatrix, translation);
    uniforms->modelMatrix = modelMatrix;
    uniforms->modelViewMatrix = matrix_multiply(uniforms->viewMatrix, modelMatrix);
    uniforms->screenWidth = _windowSize.width;
    _rotation += 0.01;
}

- (void) mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size
{
    // When reshape is called, update the aspect ratio and projection matrix since the view
    //   orientation or size has changed
    _windowSize = size;
	float aspect = size.width / (float)size.height;
    _projectionMatrix = matrix_perspective_left_hand(65.0f * (M_PI / 180.0f), aspect, 1.0f, 5000.0);
}

- (void) drawInMTKView:(nonnull MTKView *)view
{
    dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);

    id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"MyCommand";

    __block dispatch_semaphore_t block_sema = _inFlightSemaphore;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer)
    {
        dispatch_semaphore_signal(block_sema);
    }];

    [self updateGameState];

    // Obtain a renderPassDescriptor generated from the view's drawable textures
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if(renderPassDescriptor != nil) {
        MTLSize tileSize = MTLSizeMake(32, 16, 1);
        renderPassDescriptor.tileWidth = tileSize.width;
        renderPassDescriptor.tileHeight = tileSize.height;
        renderPassDescriptor.imageblockSampleLength = 64;//_resolveState.imageblockSampleLength;
        // Create a render command encoder so we can render into something
        id <MTLRenderCommandEncoder> renderEncoder =
            [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"Rendering";

        // If we're not using device memory, we need to clear the threadgroup data
        [renderEncoder pushDebugGroup:@"Clear Imageblock Memory"];
        [renderEncoder setRenderPipelineState:_clearTileState];
        [renderEncoder dispatchThreadsPerTile:tileSize];
        [renderEncoder popDebugGroup];

        // Set render command encoder state
        [renderEncoder pushDebugGroup:@"Render Mesh"];
		[renderEncoder setCullMode:MTLCullModeNone];
        [renderEncoder setDepthStencilState:_depthState];
        [renderEncoder setRenderPipelineState:_pipelineState];

        // Set our per frame buffers
        [renderEncoder setVertexBuffer:_frameUniformBuffers[_uniformBufferIndex]
                                offset:0
                               atIndex:AAPLBufferIndexFrameUniforms];

        [renderEncoder setFragmentBuffer:_frameUniformBuffers[_uniformBufferIndex]
                                  offset:0
                                 atIndex:AAPLBufferIndexFrameUniforms];

        [renderEncoder setFragmentBuffer:_oitBufferData
                                  offset:0
                                 atIndex:AAPLBufferIndexOITData];

        for (MTKMesh *metalKitMesh in _meshes)
        {
            // Set mesh's vertex buffers
            for (NSUInteger bufferIndex = 0; bufferIndex < metalKitMesh.vertexBuffers.count; bufferIndex++)
            {
                 MTKMeshBuffer *vertexBuffer = metalKitMesh.vertexBuffers[bufferIndex];
                if((NSNull*)vertexBuffer != [NSNull null])
                {
                    [renderEncoder setVertexBuffer:vertexBuffer.buffer
                                            offset:vertexBuffer.offset
                                           atIndex:bufferIndex];
                }
            }
            
            [renderEncoder setFragmentBuffer:_frameUniformBuffers[_uniformBufferIndex]
             offset:0
            atIndex:AAPLBufferIndexFrameUniforms];
            // Draw each submesh of our mesh
            [renderEncoder setFragmentTexture:_colorMap atIndex:AAPLTextureIndexBaseColor];
            for(MTKSubmesh *metalKitSubmesh in metalKitMesh.submeshes)
            {
                // Set any textures read/sampled from our render pipeline
                [renderEncoder drawIndexedPrimitives:metalKitSubmesh.primitiveType
                                          indexCount:metalKitSubmesh.indexCount
                                           indexType:metalKitSubmesh.indexType
                                         indexBuffer:metalKitSubmesh.indexBuffer.buffer
                                   indexBufferOffset:metalKitSubmesh.indexBuffer.offset];
            }
        }
        [renderEncoder popDebugGroup];
        // Resolve the OIT data from the threadgroup data
        [renderEncoder pushDebugGroup:@"ResolveTranparency"];
        [renderEncoder setRenderPipelineState:_resolveState];
        [renderEncoder dispatchThreadsPerTile:tileSize];
        [renderEncoder popDebugGroup];
        // We're done encoding commands
        [renderEncoder endEncoding];
    }

    // Schedule a present once the framebuffer is complete using the current drawable
    [commandBuffer presentDrawable:view.currentDrawable];
    // Finalize rendering here & push the command buffer to the GPU
    [commandBuffer commit];
    _frameNum++;
}

@end
