#version 330 core

in vec4 vCol;
in vec2 TexCoord;
in vec3 FragPos;
in vec3 Normal;
out vec4 colour;

uniform vec3 lightColour;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform sampler2D texture1;

vec3 ambientLight()
{
    float ambientStrength = 0.3f;
    vec3 ambient = lightColour * ambientStrength;
    return ambient;
}

// Lambertian diffuse
vec3 diffuseLight()
{
    float diffuseStrength = 0.5f;
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 norm = normalize(Normal);
    float diff = max(dot(norm, lightDir), 0.0f);
    vec3 diffuse = lightColour * diff * diffuseStrength;
    return diffuse;
}

// Phong specular
vec3 specularLight()
{
    float specularStrength = 0.8f;
    float shininess = 64.0f;
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 norm = normalize(Normal);
    vec3 reflectDir = reflect(-lightDir, norm);
    vec3 viewDir = normalize(viewPos - FragPos);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0f), shininess);
    vec3 specular = lightColour * spec * specularStrength;
    return specular;
}

void main()
{
    // Phong lighting = ambient + diffuse + specular
    colour = texture(texture1, TexCoord) *
             vec4(ambientLight() + diffuseLight() + specularLight(), 1.0f);
}
