#version 330 core
// vertex shader หลักของฉาก
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUV;
layout (location = 2) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;     // transpose(inverse(mat3(model))) คำนวณจากฝั่ง CPU
uniform mat4 lightSpace;       // projection*view ของแสงจันทร์ (ใช้ทำเงา)
uniform vec2 uvScale;          // ให้ texture ซ้ำ (tile) ได้

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out vec4 vLightSpacePos;

void main()
{
    vec4 worldPos   = model * vec4(aPos, 1.0);
    vWorldPos       = worldPos.xyz;
    vNormal         = normalMatrix * aNormal;
    vUV             = aUV * uvScale;
    vLightSpacePos  = lightSpace * worldPos;
    gl_Position     = projection * view * worldPos;
}
