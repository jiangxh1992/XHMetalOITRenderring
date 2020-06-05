/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Kernels to perform Order Independent Transparency
*/

#include <metal_stdlib>
using namespace metal;

#import "AAPLShaderCommon.h"

#define LAYERSCOUNT 6
constant uint useDeviceMemory [[function_constant(0)]];

typedef rgba8unorm<half4> rgba8storage;
typedef r8unorm<half> r8storage;

template <int NUM_LAYERS>
struct OITData
{
    static constexpr constant short s_numLayers = NUM_LAYERS;
    rgba8storage colors         [[raster_order_group(0)]] [NUM_LAYERS];
    float         depths         [[raster_order_group(0)]] [NUM_LAYERS];
    r8storage    transmittances [[raster_order_group(0)]] [NUM_LAYERS];
};

// The imageblock structure
template <int NUM_LAYERS>
struct OITImageblock
{
    OITData<NUM_LAYERS> oitData;
};

template <int NUM_LAYERS>
struct FragOut
{
    OITImageblock<NUM_LAYERS> aoitImageBlock [[imageblock_data]];
};

// OITFragmentFunction, InsertFragment, Resolve, and Clear are templatized on OITDataT   in order
//   to control the number of layers
template <typename OITDataT>
inline void InsertFragment(OITDataT oitData, half4 color, float depth, half transmittance)
{
    const short numLayers = oitData->s_numLayers;

    for (short i = 0; i < numLayers; ++i)
    {
        float layerDepth = oitData->depths[i];
        half4 layerColor = oitData->colors[i];
        half layerTransmittance = oitData->transmittances[i];

        bool insert = (abs(depth) <= abs(layerDepth));
        oitData->colors[i] = insert ? color : layerColor;
        oitData->depths[i] = insert ? depth : layerDepth;
        oitData->transmittances[i] = insert ? transmittance : layerTransmittance;

        color = insert ? layerColor : color;
        depth = insert ? layerDepth : depth;
        transmittance = insert ? layerTransmittance : transmittance;
    }
    const short lastLayer = numLayers - 1;
    float lastDepth = oitData->depths[lastLayer];
    half4 lastColor = oitData->colors[lastLayer];
    half lastTransmittance = oitData->transmittances[lastLayer];

    bool newDepthFirst = (abs(depth) <= abs(lastDepth));

    float firstDepth = newDepthFirst ? depth : lastDepth;
    half4 firstColor = newDepthFirst ? color : lastColor;
    half4 secondColor = newDepthFirst ? lastColor : color;
    half firstTransmittance = newDepthFirst ? transmittance : lastTransmittance;

    oitData->colors[lastLayer] = firstColor + secondColor * firstTransmittance;
    oitData->depths[lastLayer] = firstDepth;
    oitData->transmittances[lastLayer] = transmittance * lastTransmittance;
}


fragment FragOut<LAYERSCOUNT>
OITFragmentFunction(ColorInOut                   in            [[ stage_in ]],
                           constant AAPLFrameUniforms & uniforms      [[ buffer (AAPLBufferIndexFrameUniforms) ]],
                           texture2d<half>              baseColorMap  [[ texture(AAPLTextureIndexBaseColor) ]],
                           OITImageblock<LAYERSCOUNT>             oitImageblock [[ imageblock_data ]])
{
    float3 cameraPos = uniforms.viewMatrix.columns[3].xyz;
    float3 wposition = in.wposition.xyz;
    float3 V = normalize(cameraPos - wposition);
    float3 N = in.wnormal.xyz;
    float nv = dot(V, N);
    
    float depth = 1.0 + in.position.z;// * 65504.0f; // in.position.w;
    
    constexpr sampler linearSampler(mip_filter::linear,
                                    mag_filter::linear,
                                    min_filter::linear);
    
    half4 fragmentColor = baseColorMap.sample(linearSampler, in.texCoord);

    fragmentColor.a = 0.5;

    fragmentColor.rgb *= (1 - fragmentColor.a);
    if(nv >= 0) depth = -depth;
    InsertFragment(&oitImageblock.oitData, fragmentColor, depth, 1 - fragmentColor.a);
    
    FragOut<LAYERSCOUNT> Out;
    Out.aoitImageBlock = oitImageblock;
    return Out;
}


kernel void OITClear(imageblock<OITImageblock<LAYERSCOUNT>, imageblock_layout_explicit> oitData,
                            ushort2 tid [[thread_position_in_threadgroup]])
{
    threadgroup_imageblock OITData<LAYERSCOUNT> &pixelData = oitData.data(tid)->oitData;
    const short numLayers = pixelData.s_numLayers;
    for (ushort i = 0; i < numLayers; ++i)
    {
        pixelData.colors[i] = half4(0.0);
        pixelData.depths[i] = 2.1f;
        pixelData.transmittances[i] = 1.0;
    }
}

fragment half4 OITResolve(OITImageblock<LAYERSCOUNT> oitImageblock [[imageblock_data]])
{
    half4 rtColor = half4(0,0,0,1.0);
    OITData<LAYERSCOUNT> pixelData = oitImageblock.oitData;
    const short numLayers = pixelData.s_numLayers;

    // Composite!
    half4 finalColor = rtColor;
    half transmittance = 1;
    /*
    // 厚度计算
    float thickness = 0;
    int left = 1;
    for( int i = 0, j = 1; ; )
    {
        // 就位
        while( pixelData.depths[i] <= 0 && i < numLayers ) { ++i; ++j;}
        if( i >= numLayers || j >= numLayers) break;
    
        // 搜索对应反面j
        while( pixelData.depths[j] > 0 || left > 1 )
        {
            if( pixelData.depths[j] > 0 ) ++left;
            else if( left > 1 )
            {
            --left;
            }
            ++j;
    
            if(j > numLayers) break;
        }
    
        // 计算当前厚度
        if(j <= numLayers)
        {
        thickness += (abs(pixelData.depths[j]) - abs(pixelData.depths[i]));
        }
    
        // 重置
        i = j + 1;
        j = j + 2;
    }
    */
    //thickness = (abs(pixelData.depths[1]) - abs(pixelData.depths[0]));
    
    // 透射
    //thickness *= 10.0f;
    //float opaque = exp(-thickness);
    //finalColor = (half4)pixelData.colors[0] * opaque;
    for (ushort i = 0; i < numLayers; ++i)
    {
        transmittance *= (half)pixelData.transmittances[i];
        finalColor += (half4)pixelData.colors[i] * transmittance;
    }
    finalColor.w = 1;
    return finalColor;
}
