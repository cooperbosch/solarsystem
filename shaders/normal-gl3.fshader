#version 150

uniform sampler2D uTexColor;
uniform sampler2D uTexNormal;

// lights in eye space
uniform vec3 uLight;

in vec2 vTexCoord;
in mat3 vNTMat;
in vec3 vEyePos;

out vec4 fragColor;

void main() {
  vec3 normal = texture(uTexNormal, vTexCoord).xyz * 2.0 - 1.0;

  normal = normalize(vNTMat * normal);

  vec3 viewDir = normalize(-vEyePos);
  vec3 lightDir = normalize(uLight - vEyePos);

  float nDotL = dot(normal, lightDir);
  vec3 reflection = normalize( 2.0 * normal * nDotL - lightDir);
  float rDotV = max(0.0, dot(reflection, viewDir));
  float specular = pow(rDotV, 32.0);
  float diffuse = max(nDotL, 0.0);

  vec3 color = texture(uTexColor, vTexCoord).xyz + specular * vec3(0.6, 0.6, 0.6);

  fragColor = vec4(color, 1);
}
