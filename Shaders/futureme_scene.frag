#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec4 vLightSpacePos;

out vec4 FragColor;

// material
uniform sampler2D albedoTex;
uniform vec3  albedo;          // สี
uniform float alpha;
uniform float specStrength;
uniform float shininess;
uniform vec3  emissive;        // แสงที่วัตถุปล่อยเอง
uniform float emissiveTex;     // > 0 = ให้ texture เรืองแสง (เช่นหน้าจอ)
uniform bool  doubleSided;     // วัตถุบางๆ ที่มองได้สองด้าน (ใยแมงมุม, โพสต์อิท)

// camera
uniform vec3 viewPos;

// ambient
uniform vec3 ambientSky;
uniform vec3 ambientGround;

// directional light (moon)
uniform vec3 moonDir;          // ทิศที่แสงเดินทาง
uniform vec3 moonColor;
uniform sampler2D shadowMap;

// point lights
#define MAX_POINT_LIGHTS 4
uniform int  numPointLights;
uniform vec3 pointPos[MAX_POINT_LIGHTS];
uniform vec3 pointColor[MAX_POINT_LIGHTS];
uniform vec3 pointDir[MAX_POINT_LIGHTS];
uniform vec3 pointAtt[MAX_POINT_LIGHTS];   // (constant, linear, quadratic)

// post
uniform vec2  resolution;
uniform float exposure;

vec3 toLinear(vec3 c) { return pow(c, vec3(2.2)); }

float ShadowFactor(vec3 N, vec3 L)
{
    vec3 p = vLightSpacePos.xyz / vLightSpacePos.w;
    p = p * 0.5 + 0.5;
    if (p.z > 1.0 || p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return 0.0;

    float bias = max(0.0030 * (1.0 - dot(N, L)), 0.0007);
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    float shadow = 0.0;
    // PCF 5x5 ให้ขอบเงานุ่มขึ้น
    for (int x = -2; x <= 2; ++x)
        for (int y = -2; y <= 2; ++y)
        {
            float d = texture(shadowMap, p.xy + vec2(x, y) * texel * 1.5).r;
            shadow += (p.z - bias > d) ? 1.0 : 0.0;
        }
    return shadow / 25.0;
}

vec3 ACESFilm(vec3 x)
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    vec4 texel = texture(albedoTex, vUV);
    if (texel.a < 0.1) discard;               
    vec3 base  = toLinear(albedo) * toLinear(texel.rgb);
    float a    = alpha * texel.a;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(viewPos - vWorldPos);
    if (doubleSided && dot(N, V) < 0.0) N = -N;

    // ambient
    vec3 colour = base * mix(toLinear(ambientGround), toLinear(ambientSky), N.y * 0.5 + 0.5);

    // moon light
    vec3 L = normalize(-moonDir);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL > 0.0)
    {
        float shadow = ShadowFactor(N, L);
        vec3  H      = normalize(L + V);
        float spec   = pow(max(dot(N, H), 0.0), shininess) * specStrength;
        colour += (base * NdotL + vec3(spec) * NdotL) * moonColor * (1.0 - shadow);
    }

    // point lights
    for (int i = 0; i < numPointLights; ++i)
    {
        vec3  Lv   = pointPos[i] - vWorldPos;
        float dist = length(Lv);
        vec3  Lp   = Lv / dist;
        float att  = 1.0 / (pointAtt[i].x + pointAtt[i].y * dist + pointAtt[i].z * dist * dist);

        float lobe = 1.0;
        if (dot(pointDir[i], pointDir[i]) > 0.0)        // แสงจากจอ ส่องไปด้านหน้าเท่านั้น
            lobe = smoothstep(-0.15, 0.6, dot(-Lp, pointDir[i]));

        float diff = max(dot(N, Lp), 0.0);
        vec3  H    = normalize(Lp + V);
        float spec = pow(max(dot(N, H), 0.0), shininess) * specStrength;
        colour += (base * diff + vec3(spec) * diff) * pointColor[i] * att * lobe;
    }

    // emissive 
    colour += toLinear(emissive) + base * emissiveTex;

    // tone mapping + gamma
    colour = ACESFilm(colour * exposure);
    colour = pow(colour, vec3(1.0 / 2.2));

    // vignette
    vec2 uv = gl_FragCoord.xy / resolution;
    float vig = smoothstep(0.85, 0.30, length(uv - 0.5));
    colour *= mix(0.45, 1.0, vig);

    FragColor = vec4(colour, a);
}
