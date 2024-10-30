#version 150

uniform vec3 uLight, uColor;

in vec3 vNormal;
in vec3 vPosition;

out vec4 fragColor;

void main() {
  vec3 tolight = normalize(uLight - vPosition);
  vec3 normal = normalize(vNormal);

  float diffuse = max(0.0, dot(normal, tolight));
  vec3 intensity = uColor * diffuse;

  fragColor = vec4(intensity, 1.0);
}
