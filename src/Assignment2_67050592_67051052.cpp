// =====================================================================================
//  Assignment 2 : "FUTURE ME"
//  ภาพ: โปรแกรมเมอร์นั่งรอโปรแกรมรันนานมาก ... จนกลายเป็นโครงกระดูก แล้วสุดท้ายก็ขึ้น ERROR
//
//  - ทุกชิ้นในฉากสร้างจาก vertices ด้วยโค้ด (box / sphere / tube / lathe) ไม่ใช้ไฟล์โมเดลภายนอก
//  - Texture ทั้งหมด (ไม้, พื้น, ผนัง, หน้าจอ error, ปฏิทิน, โปสเตอร์ ...) สร้างด้วยโค้ดเช่นกัน
//  - Modern OpenGL 3.3 core + Lighting model แบบ Blinn-Phong
//        * แสงจันทร์ (directional) ลอดหน้าต่าง พร้อม shadow mapping
//        * แสงสีแดงจากหน้าจอที่ขึ้น error (point light), ไฟ LED เคสคอม
//        * ambient แบบ hemisphere, emissive, tone mapping, vignette
//  - เรนเดอร์ภาพขนาด 800 x 600 (MSAA 4x)
//
//  ปุ่มควบคุม
//     P              : บันทึกภาพ 800x600 เป็น PNG (ชื่อไฟล์ตั้งได้ที่ SCREENSHOT_NAME)
//     คลิกขวาค้าง + ลากเมาส์ : หมุนมุมกล้อง
//     W A S D / Q E  : เลื่อนกล้อง (หน้า/หลัง/ซ้าย/ขวา/ลง/ขึ้น)
//     R              : กลับไปมุมกล้องหลักของภาพ
//     ESC            : ออกจากโปรแกรม
//
//  ไฟล์ shader ที่ใช้ (อยู่ในโฟลเดอร์ Shaders/)
//     futureme_scene.vert, futureme_scene.frag, futureme_shadow.vert, futureme_shadow.frag
// =====================================================================================
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#undef GLFW_DLL

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Libs/Mesh.h"
#include "Libs/Shader.h"
#include "Libs/Window.h"
#include "ProjectPaths.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

// ------------------------------------------------------------------------------------
//  Settings
// ------------------------------------------------------------------------------------
const GLint WIDTH  = 800;
const GLint HEIGHT = 600;
const int   SHADOW_SIZE = 2048;
const float PI = 3.14159265358979f;

// เปลี่ยนชื่อไฟล์ภาพตามรหัสนักศึกษา เช่น "Assignment2_67050001_67050002.png"
const char* SCREENSHOT_NAME = "Assignment2_67051052_yourPairID.png";

const std::filesystem::path shaderDirectory =
    std::filesystem::u8path(OPENGL_STARTER_SHADER_DIR);

// ------------------------------------------------------------------------------------
//  Math helpers
// ------------------------------------------------------------------------------------
static const vec3 X_AXIS(1.0f, 0.0f, 0.0f);
static const vec3 Y_AXIS(0.0f, 1.0f, 0.0f);
static const vec3 Z_AXIS(0.0f, 0.0f, 1.0f);

static mat4 T(const vec3& t) { return glm::translate(mat4(1.0f), t); }
static mat4 S(const vec3& s) { return glm::scale(mat4(1.0f), s); }
static mat4 S(float s) { return glm::scale(mat4(1.0f), vec3(s)); }
static mat4 R(float deg, const vec3& axis) { return glm::rotate(mat4(1.0f), glm::radians(deg), axis); }
static vec3 xform(const mat4& M, const vec3& p) { return vec3(M * vec4(p, 1.0f)); }
static vec3 xdir(const mat4& M, const vec3& d) { return vec3(M * vec4(d, 0.0f)); }
static vec3 safeNormalize(const vec3& v)
{
    float l = glm::length(v);
    return (l > 1e-8f) ? v / l : vec3(0.0f, 1.0f, 0.0f);
}
static vec3 lerp3(const vec3& a, const vec3& b, float t) { return a + (b - a) * t; }

// ------------------------------------------------------------------------------------
//  MeshBuilder : รวม primitive หลายๆ ชิ้น (ที่ใช้วัสดุเดียวกัน) ให้เป็น Mesh เดียว
//  ทุกฟังก์ชันสร้าง vertex (position, normal, uv) เองทั้งหมด
// ------------------------------------------------------------------------------------
static VertexPNU MakeVertex(const vec3& p, const vec3& n, const vec2& uv)
{
    VertexPNU v;
    v.position = p;
    v.normal = n;
    v.uv = uv;
    return v;
}

struct MeshBuilder
{
    std::vector<VertexPNU> verts;
    std::vector<unsigned int> idx;

    unsigned int Base() const { return static_cast<unsigned int>(verts.size()); }

    // กล่องขนาด 1x1x1 (กึ่งกลางที่ origin) แล้วแปลงด้วย M
    void Box(const mat4& M)
    {
        const mat3 NM = glm::transpose(glm::inverse(mat3(M)));
        const vec3 normals[6] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1} };
        for (int f = 0; f < 6; ++f)
        {
            vec3 N = normals[f];
            vec3 U = (std::fabs(N.y) > 0.5f) ? vec3(0.0f, 0.0f, -1.0f) : vec3(0.0f, 1.0f, 0.0f);
            vec3 A = glm::cross(U, N);      // แกน u ของหน้า
            vec3 B = glm::cross(N, A);      // แกน v ของหน้า
            vec3 C = N * 0.5f;
            unsigned int b = Base();
            vec3 wn = safeNormalize(NM * N);
            verts.push_back(MakeVertex(xform(M, C - A * 0.5f - B * 0.5f), wn, vec2(0, 0)));
            verts.push_back(MakeVertex(xform(M, C + A * 0.5f - B * 0.5f), wn, vec2(1, 0)));
            verts.push_back(MakeVertex(xform(M, C + A * 0.5f + B * 0.5f), wn, vec2(1, 1)));
            verts.push_back(MakeVertex(xform(M, C - A * 0.5f + B * 0.5f), wn, vec2(0, 1)));
            idx.insert(idx.end(), { b, b + 1, b + 2, b, b + 2, b + 3 });
        }
    }

    // ทรงกลมรัศมี 1 แล้วแปลงด้วย M (ใช้ scale ไม่เท่ากันเพื่อทำทรงรี)
    void Sphere(const mat4& M, int slices = 20, int stacks = 14)
    {
        const mat3 NM = glm::transpose(glm::inverse(mat3(M)));
        unsigned int b = Base();
        for (int i = 0; i <= stacks; ++i)
        {
            float phi = PI * i / stacks;
            for (int j = 0; j <= slices; ++j)
            {
                float th = 2.0f * PI * j / slices;
                vec3 p(std::sin(phi) * std::cos(th), std::cos(phi), std::sin(phi) * std::sin(th));
                verts.push_back(MakeVertex(xform(M, p), safeNormalize(NM * p),
                                           vec2((float)j / slices, 1.0f - (float)i / stacks)));
            }
        }
        for (int i = 0; i < stacks; ++i)
            for (int j = 0; j < slices; ++j)
            {
                unsigned int a = b + i * (slices + 1) + j;
                unsigned int c = a + slices + 1;
                idx.insert(idx.end(), { a, a + 1, c, a + 1, c + 1, c });
            }
    }

    // ท่อไปตามเส้นทาง pts (รัศมีแต่ละจุดตาม radii) ใช้ parallel-transport frame
    void Tube(const std::vector<vec3>& pts, const std::vector<float>& radii, int slices = 10, bool caps = true)
    {
        const size_t n = pts.size();
        if (n < 2) return;
        std::vector<vec3> Tn(n), Nn(n), Bn(n);
        for (size_t i = 0; i < n; ++i)
        {
            vec3 a = pts[i > 0 ? i - 1 : 0];
            vec3 c = pts[i + 1 < n ? i + 1 : n - 1];
            Tn[i] = safeNormalize(c - a);
        }
        vec3 up = (std::fabs(Tn[0].y) < 0.9f) ? Y_AXIS : X_AXIS;
        Nn[0] = safeNormalize(glm::cross(Tn[0], up));
        Bn[0] = glm::cross(Tn[0], Nn[0]);
        for (size_t i = 1; i < n; ++i)
        {
            vec3 v = Nn[i - 1] - Tn[i] * glm::dot(Nn[i - 1], Tn[i]);
            Nn[i] = safeNormalize(v);
            Bn[i] = glm::cross(Tn[i], Nn[i]);
        }

        unsigned int b = Base();
        for (size_t i = 0; i < n; ++i)
        {
            size_t ia = (i > 0) ? i - 1 : 0, ic = (i + 1 < n) ? i + 1 : n - 1;
            float segLen = glm::length(pts[ic] - pts[ia]);
            float slope = (segLen > 1e-6f) ? (radii[ic] - radii[ia]) / segLen : 0.0f;
            for (int j = 0; j <= slices; ++j)
            {
                float th = 2.0f * PI * j / slices;
                vec3 dir = Nn[i] * std::cos(th) + Bn[i] * std::sin(th);
                vec3 nrm = safeNormalize(dir - Tn[i] * slope);
                verts.push_back(MakeVertex(pts[i] + dir * radii[i], nrm,
                                           vec2((float)j / slices, (float)i / (n - 1))));
            }
        }
        for (size_t i = 0; i + 1 < n; ++i)
            for (int j = 0; j < slices; ++j)
            {
                unsigned int a = b + (unsigned int)(i * (slices + 1) + j);
                unsigned int c = a + slices + 1;
                idx.insert(idx.end(), { a, c, a + 1, a + 1, c, c + 1 });
            }

        if (caps)
        {
            for (int e = 0; e < 2; ++e)
            {
                size_t i = (e == 0) ? 0 : n - 1;
                vec3 nrm = (e == 0) ? -Tn[i] : Tn[i];
                unsigned int centre = Base();
                verts.push_back(MakeVertex(pts[i], nrm, vec2(0.5f, 0.5f)));
                for (int j = 0; j <= slices; ++j)
                {
                    float th = 2.0f * PI * j / slices;
                    vec3 dir = Nn[i] * std::cos(th) + Bn[i] * std::sin(th);
                    verts.push_back(MakeVertex(pts[i] + dir * radii[i], nrm,
                                               vec2(0.5f + 0.5f * std::cos(th), 0.5f + 0.5f * std::sin(th))));
                }
                for (int j = 0; j < slices; ++j)
                    idx.insert(idx.end(), { centre, centre + 1 + j, centre + 2 + j });
            }
        }
    }

    // ทรงกระบอก/กรวยระหว่างจุด a -> b
    void Cylinder(const vec3& a, const vec3& b, float ra, float rb, int slices = 14, bool caps = true)
    {
        Tube({ a, b }, { ra, rb }, slices, caps);
    }

    // พื้นผิวหมุนรอบแกน y (lathe) : profile = (รัศมี, ความสูง) จากล่างขึ้นบน
    void Lathe(const std::vector<vec2>& profile, const mat4& M, int slices = 32)
    {
        const mat3 NM = glm::transpose(glm::inverse(mat3(M)));
        const size_t n = profile.size();
        // v ของ texture = ระยะทางสะสมตาม profile (texture จะไม่ยืดผิดสัดส่วน)
        std::vector<float> arc(n, 0.0f);
        for (size_t i = 1; i < n; ++i) arc[i] = arc[i - 1] + glm::length(profile[i] - profile[i - 1]);
        unsigned int b = Base();
        for (size_t i = 0; i < n; ++i)
        {
            vec2 pa = profile[i > 0 ? i - 1 : 0];
            vec2 pc = profile[i + 1 < n ? i + 1 : n - 1];
            vec2 t = pc - pa;
            vec2 n2 = glm::normalize(vec2(t.y, -t.x));     // normal ชี้ออกด้านนอก
            for (int j = 0; j <= slices; ++j)
            {
                float th = 2.0f * PI * j / slices;
                float c = std::cos(th), s = std::sin(th);
                vec3 p(profile[i].x * c, profile[i].y, profile[i].x * s);
                vec3 nn(n2.x * c, n2.y, n2.x * s);
                verts.push_back(MakeVertex(xform(M, p), safeNormalize(NM * nn),
                                           vec2(1.0f - (float)j / slices, arc[i] / arc[n - 1])));
            }
        }
        for (size_t i = 0; i + 1 < n; ++i)
            for (int j = 0; j < slices; ++j)
            {
                unsigned int a = b + (unsigned int)(i * (slices + 1) + j);
                unsigned int c = a + slices + 1;
                idx.insert(idx.end(), { a, c, a + 1, a + 1, c, c + 1 });
            }
    }

    // สี่เหลี่ยมแบน 1x1 บนระนาบ XY หันหน้าไป +Z
    void Quad(const mat4& M)
    {
        const mat3 NM = glm::transpose(glm::inverse(mat3(M)));
        vec3 n = safeNormalize(NM * Z_AXIS);
        unsigned int b = Base();
        verts.push_back(MakeVertex(xform(M, vec3(-0.5f, -0.5f, 0.0f)), n, vec2(0, 0)));
        verts.push_back(MakeVertex(xform(M, vec3( 0.5f, -0.5f, 0.0f)), n, vec2(1, 0)));
        verts.push_back(MakeVertex(xform(M, vec3( 0.5f,  0.5f, 0.0f)), n, vec2(1, 1)));
        verts.push_back(MakeVertex(xform(M, vec3(-0.5f,  0.5f, 0.0f)), n, vec2(0, 1)));
        idx.insert(idx.end(), { b, b + 1, b + 2, b, b + 2, b + 3 });
    }

    Mesh* Build() const
    {
        Mesh* m = new Mesh();
        m->CreateMesh(verts, idx);
        return m;
    }
};

// ------------------------------------------------------------------------------------
//  Procedural noise (ใช้ทำลายไม้ พื้น ผนัง)
// ------------------------------------------------------------------------------------
static float Hash2(int x, int y, int seed)
{
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)seed * 2246822519u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (h & 0xFFFFFFu) / 16777215.0f;
}

static float ValueNoise(float x, float y, int seed, int period)
{
    int xi = (int)std::floor(x), yi = (int)std::floor(y);
    float fx = x - xi, fy = y - yi;
    auto wrap = [period](int a) { return ((a % period) + period) % period; };
    float a = Hash2(wrap(xi), wrap(yi), seed);
    float b = Hash2(wrap(xi + 1), wrap(yi), seed);
    float c = Hash2(wrap(xi), wrap(yi + 1), seed);
    float d = Hash2(wrap(xi + 1), wrap(yi + 1), seed);
    float ux = fx * fx * (3.0f - 2.0f * fx), uy = fy * fy * (3.0f - 2.0f * fy);
    return glm::mix(glm::mix(a, b, ux), glm::mix(c, d, ux), uy);
}

// fractal noise ที่ต่อกันได้ไร้รอยต่อ (tileable) : u,v อยู่ในช่วง 0..1
static float Fbm(float u, float v, int seed, int period, int octaves = 5)
{
    float sum = 0.0f, amp = 0.5f, norm = 0.0f;
    int p = period;
    for (int o = 0; o < octaves; ++o)
    {
        sum += amp * ValueNoise(u * p, v * p, seed + o * 31, p);
        norm += amp;
        amp *= 0.5f;
        p *= 2;
    }
    return sum / norm;
}

// ------------------------------------------------------------------------------------
//  Bitmap font 5x7 (ออกแบบเอง) สำหรับเขียนตัวอักษรลง texture
// ------------------------------------------------------------------------------------
struct Glyph { char c; const char* rows; };
static const Glyph FONT[] = {
    {'A', ".###." "#...#" "#...#" "#####" "#...#" "#...#" "#...#"},
    {'B', "####." "#...#" "#...#" "####." "#...#" "#...#" "####."},
    {'C', ".###." "#...#" "#...." "#...." "#...." "#...#" ".###."},
    {'D', "###.." "#..#." "#...#" "#...#" "#...#" "#..#." "###.."},
    {'E', "#####" "#...." "#...." "####." "#...." "#...." "#####"},
    {'F', "#####" "#...." "#...." "####." "#...." "#...." "#...."},
    {'G', ".###." "#...#" "#...." "#.###" "#...#" "#...#" ".####"},
    {'H', "#...#" "#...#" "#...#" "#####" "#...#" "#...#" "#...#"},
    {'I', ".###." "..#.." "..#.." "..#.." "..#.." "..#.." ".###."},
    {'J', "..###" "...#." "...#." "...#." "...#." "#..#." ".##.."},
    {'K', "#...#" "#..#." "#.#.." "##..." "#.#.." "#..#." "#...#"},
    {'L', "#...." "#...." "#...." "#...." "#...." "#...." "#####"},
    {'M', "#...#" "##.##" "#.#.#" "#.#.#" "#...#" "#...#" "#...#"},
    {'N', "#...#" "#...#" "##..#" "#.#.#" "#..##" "#...#" "#...#"},
    {'O', ".###." "#...#" "#...#" "#...#" "#...#" "#...#" ".###."},
    {'P', "####." "#...#" "#...#" "####." "#...." "#...." "#...."},
    {'Q', ".###." "#...#" "#...#" "#...#" "#.#.#" "#..#." ".##.#"},
    {'R', "####." "#...#" "#...#" "####." "#.#.." "#..#." "#...#"},
    {'S', ".####" "#...." "#...." ".###." "....#" "....#" "####."},
    {'T', "#####" "..#.." "..#.." "..#.." "..#.." "..#.." "..#.."},
    {'U', "#...#" "#...#" "#...#" "#...#" "#...#" "#...#" ".###."},
    {'V', "#...#" "#...#" "#...#" "#...#" "#...#" ".#.#." "..#.."},
    {'W', "#...#" "#...#" "#...#" "#.#.#" "#.#.#" "#.#.#" ".#.#."},
    {'X', "#...#" "#...#" ".#.#." "..#.." ".#.#." "#...#" "#...#"},
    {'Y', "#...#" "#...#" ".#.#." "..#.." "..#.." "..#.." "..#.."},
    {'Z', "#####" "....#" "...#." "..#.." ".#..." "#...." "#####"},
    {'0', ".###." "#...#" "#..##" "#.#.#" "##..#" "#...#" ".###."},
    {'1', "..#.." ".##.." "..#.." "..#.." "..#.." "..#.." ".###."},
    {'2', ".###." "#...#" "....#" "...#." "..#.." ".#..." "#####"},
    {'3', "#####" "...#." "..#.." "...#." "....#" "#...#" ".###."},
    {'4', "...#." "..##." ".#.#." "#..#." "#####" "...#." "...#."},
    {'5', "#####" "#...." "####." "....#" "....#" "#...#" ".###."},
    {'6', "..##." ".#..." "#...." "####." "#...#" "#...#" ".###."},
    {'7', "#####" "....#" "...#." "..#.." ".#..." ".#..." ".#..."},
    {'8', ".###." "#...#" "#...#" ".###." "#...#" "#...#" ".###."},
    {'9', ".###." "#...#" "#...#" ".####" "....#" "...#." ".##.."},
    {'.', "....." "....." "....." "....." "....." ".##.." ".##.."},
    {',', "....." "....." "....." "....." ".##.." "..#.." ".#..."},
    {':', "....." ".##.." ".##.." "....." ".##.." ".##.." "....."},
    {'!', "..#.." "..#.." "..#.." "..#.." "..#.." "....." "..#.."},
    {'?', ".###." "#...#" "....#" "...#." "..#.." "....." "..#.."},
    {'-', "....." "....." "....." "#####" "....." "....." "....."},
    {'_', "....." "....." "....." "....." "....." "....." "#####"},
    {'/', "....." "....#" "...#." "..#.." ".#..." "#...." "....."},
    {'\\',"....." "#...." ".#..." "..#.." "...#." "....#" "....."},
    {'(', "...#." "..#.." ".#..." ".#..." ".#..." "..#.." "...#."},
    {')', ".#..." "..#.." "...#." "...#." "...#." "..#.." ".#..."},
    {'[', ".###." ".#..." ".#..." ".#..." ".#..." ".#..." ".###."},
    {']', ".###." "...#." "...#." "...#." "...#." "...#." ".###."},
    {'#', ".#.#." ".#.#." "#####" ".#.#." "#####" ".#.#." ".#.#."},
    {'%', "##..." "##..#" "...#." "..#.." ".#..." "#..##" "...##"},
    {'>', ".#..." "..#.." "...#." "....#" "...#." "..#.." ".#..."},
    {'<', "...#." "..#.." ".#..." "#...." ".#..." "..#.." "...#."},
    {'=', "....." "....." "#####" "....." "#####" "....." "....."},
    {'+', "....." "..#.." "..#.." "#####" "..#.." "..#.." "....."},
    {'*', "....." "#.#.#" ".###." "#####" ".###." "#.#.#" "....."},
    {'\'',"..#.." "..#.." ".#..." "....." "....." "....." "....."},
    {'"', ".#.#." ".#.#." ".#.#." "....." "....." "....." "....."},
    {'|', "..#.." "..#.." "..#.." "..#.." "..#.." "..#.." "..#.."},
    {'@', "#####" "#####" "#####" "#####" "#####" "#####" "#####"},   // '@' = บล็อกทึบ (ใช้ทำ progress bar)
};

static const char* FindGlyph(char c)
{
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    for (const Glyph& g : FONT)
        if (g.c == c) return g.rows;
    return nullptr;
}

// ------------------------------------------------------------------------------------
//  Canvas : วาดรูป 2D ลง buffer แล้วอัปโหลดเป็น texture
// ------------------------------------------------------------------------------------
struct Canvas
{
    int w, h;
    std::vector<float> px;   // RGBA 0..1 (แถว 0 = ด้านบนของภาพ)

    Canvas(int width, int height, const vec3& c, float a = 1.0f) : w(width), h(height), px(width * height * 4)
    {
        for (int i = 0; i < w * h; ++i)
        {
            px[i * 4 + 0] = c.x; px[i * 4 + 1] = c.y; px[i * 4 + 2] = c.z; px[i * 4 + 3] = a;
        }
    }

    void Set(int x, int y, const vec3& c, float a = 1.0f)
    {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        float* p = &px[(y * w + x) * 4];
        p[0] = glm::mix(p[0], c.x, a);
        p[1] = glm::mix(p[1], c.y, a);
        p[2] = glm::mix(p[2], c.z, a);
        p[3] = std::max(p[3], a);
    }
    vec3 Get(int x, int y) const
    {
        const float* p = &px[(y * w + x) * 4];
        return vec3(p[0], p[1], p[2]);
    }
    void Rect(int x0, int y0, int x1, int y1, const vec3& c, float a = 1.0f)
    {
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) Set(x, y, c, a);
    }
    void Frame(int x0, int y0, int x1, int y1, int t, const vec3& c)
    {
        Rect(x0, y0, x1, y0 + t, c); Rect(x0, y1 - t, x1, y1, c);
        Rect(x0, y0, x0 + t, y1, c); Rect(x1 - t, y0, x1, y1, c);
    }
    void Disc(float cx, float cy, float r, const vec3& c, float a = 1.0f)
    {
        for (int y = (int)(cy - r - 2); y <= (int)(cy + r + 2); ++y)
            for (int x = (int)(cx - r - 2); x <= (int)(cx + r + 2); ++x)
            {
                float d = std::sqrt((x + 0.5f - cx) * (x + 0.5f - cx) + (y + 0.5f - cy) * (y + 0.5f - cy));
                float cov = glm::clamp(r - d + 0.5f, 0.0f, 1.0f);
                if (cov > 0.0f) Set(x, y, c, a * cov);
            }
    }
    void Line(float x0, float y0, float x1, float y1, float thick, const vec3& c, float a = 1.0f)
    {
        float len = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        int steps = std::max(1, (int)(len * 2));
        for (int i = 0; i <= steps; ++i)
        {
            float t = (float)i / steps;
            Disc(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, thick * 0.5f, c, a);
        }
    }
    static int TextWidth(const std::string& s, int scale) { return (int)s.size() * 6 * scale - scale; }
    void Text(int x, int y, const std::string& s, int scale, const vec3& c, float a = 1.0f)
    {
        int cx = x;
        for (char ch : s)
        {
            const char* g = FindGlyph(ch);
            if (g)
                for (int r = 0; r < 7; ++r)
                    for (int k = 0; k < 5; ++k)
                        if (g[r * 5 + k] == '#') Rect(cx + k * scale, y + r * scale, cx + (k + 1) * scale, y + (r + 1) * scale, c, a);
            cx += 6 * scale;
        }
    }
    void TextCentered(int cx, int y, const std::string& s, int scale, const vec3& c)
    {
        Text(cx - TextWidth(s, scale) / 2, y, s, scale, c);
    }

    GLuint Upload(bool repeat = true) const
    {
        std::vector<unsigned char> data(w * h * 4);
        for (int y = 0; y < h; ++y)              // พลิกแนวตั้ง: OpenGL เริ่มแถวล่างสุด
            for (int x = 0; x < w * 4; ++x)
                data[(y * w * 4) + x] = (unsigned char)(glm::clamp(px[((h - 1 - y) * w * 4) + x], 0.0f, 1.0f) * 255.0f + 0.5f);
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        return tex;
    }
};

// ------------------------------------------------------------------------------------
//  Procedural textures
// ------------------------------------------------------------------------------------
struct Textures
{
    GLuint white = 0, wood = 0, floor = 0, wall = 0, fabric = 0, screen = 0, calendar = 0,
           poster = 0, sticky1 = 0, sticky2 = 0, clock = 0, can = 0, paper = 0;
} gTex;

static GLuint MakeWoodTexture()          // ลายไม้โต๊ะ (เสี้ยนไม้วิ่งตามแกน u)
{
    Canvas c(512, 256, vec3(0));
    for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
        {
            float u = (float)x / c.w, v = (float)y / c.h;
            float warp = Fbm(u, v, 11, 3);
            float t = v * 9.0f + warp * 3.5f;
            float ring = 0.5f + 0.5f * std::sin(t * 2.0f * PI);
            ring = std::pow(ring, 2.5f);
            float streak = ValueNoise(u * 4.0f, v * 90.0f, 5, 1000) * 0.5f + ValueNoise(u * 16.0f, v * 160.0f, 9, 1000) * 0.5f;
            vec3 light(0.58f, 0.40f, 0.24f), dark(0.33f, 0.20f, 0.11f);
            vec3 col = glm::mix(light, dark, glm::clamp(ring * 0.65f + streak * 0.35f, 0.0f, 1.0f));
            c.Set(x, y, col);
        }
    return c.Upload();
}

static GLuint MakeFloorTexture()         // พื้นไม้ลามิเนตเก่าๆ มีฝุ่น
{
    Canvas c(512, 512, vec3(0));
    const int plankH = 64, plankL = 256;
    for (int y = 0; y < c.h; ++y)
    {
        int row = y / plankH;
        int offset = (int)(Hash2(row, 3, 77) * plankL);
        for (int x = 0; x < c.w; ++x)
        {
            int xx = (x + offset) % c.w;
            int plank = xx / plankL;
            float tint = 0.80f + 0.35f * Hash2(row, plank, 13);
            float u = (float)x / c.w, v = (float)y / c.h;
            float grain = 0.5f + 0.5f * std::sin(((y % plankH) / (float)plankH * 5.0f + Fbm(u, v, 21, 4) * 4.0f) * 2.0f * PI);
            float fine = ValueNoise(u * 8.0f, v * 220.0f, 3, 1000);
            vec3 col = glm::mix(vec3(0.36f, 0.25f, 0.17f), vec3(0.24f, 0.16f, 0.10f), grain * 0.6f + fine * 0.4f) * tint;
            float dust = Fbm(u, v, 99, 4);
            col = glm::mix(col, vec3(0.42f, 0.40f, 0.37f), glm::clamp((dust - 0.45f) * 1.2f, 0.0f, 0.35f));
            if (y % plankH < 2 || xx % plankL < 2) col *= 0.35f;    // ร่องระหว่างแผ่นไม้
            c.Set(x, y, col);
        }
    }
    return c.Upload();
}

static GLuint MakeWallTexture()          // ผนังวอลเปเปอร์ลายทางซีดๆ มีคราบ
{
    Canvas c(256, 256, vec3(0));
    for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
        {
            float u = (float)x / c.w, v = (float)y / c.h;
            float n = Fbm(u, v, 41, 4);
            float stain = Fbm(u, v, 57, 2);
            float stripe = 0.5f + 0.5f * std::sin(u * 2.0f * PI * 8.0f);
            vec3 col = glm::mix(vec3(0.38f, 0.43f, 0.40f), vec3(0.33f, 0.38f, 0.36f), stripe);
            col *= 0.85f + 0.3f * n;
            col *= 1.0f - glm::clamp((stain - 0.55f) * 1.5f, 0.0f, 0.3f);
            c.Set(x, y, col);
        }
    return c.Upload();
}

static GLuint MakeFabricTexture()        // ผ้าเก้าอี้สำนักงาน
{
    Canvas c(128, 128, vec3(0));
    for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
        {
            float weave = ((x / 2 + y / 2) % 2 == 0) ? 1.0f : 0.82f;
            float n = 0.85f + 0.3f * Hash2(x, y, 5);
            c.Set(x, y, vec3(0.16f, 0.17f, 0.20f) * weave * n);
        }
    return c.Upload();
}

static GLuint MakePaperTexture()
{
    Canvas c(64, 64, vec3(0));
    for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
            c.Set(x, y, vec3(0.86f, 0.84f, 0.78f) * (0.94f + 0.06f * Hash2(x, y, 8)));
    return c.Upload();
}

// ---------------- หน้าจอคอมพิวเตอร์ที่ขึ้น ERROR ----------------
const int SCREEN_TEX_W = 768, SCREEN_TEX_H = 432;
static int gPromptEndX = 0, gPromptY = 0;   // ตำแหน่ง cursor กะพริบ (หน่วย pixel ของ texture)

static GLuint MakeScreenTexture()
{
    Canvas c(SCREEN_TEX_W, SCREEN_TEX_H, vec3(0.015f, 0.02f, 0.035f));
    const vec3 green(0.45f, 1.0f, 0.55f), grey(0.70f, 0.75f, 0.78f), white(1.0f), red(0.95f, 0.12f, 0.10f);

    c.Text(14, 10, "C:\\FUTURE_ME> G++ MAIN.CPP -O ME.EXE", 3, green);
    c.Text(14, 34, "C:\\FUTURE_ME> ME.EXE", 3, green);
    c.Text(14, 58, "RUNNING [@@@@@@@@@@@@@@@@@@@@@@@@.] 99%", 3, grey);
    c.Text(14, 82, "WAITED: 60 YEARS 4 MONTHS 2 DAYS", 3, grey);

    // หน้าต่าง error
    const int x0 = 70, y0 = 116, x1 = 698, y1 = 388;
    c.Rect(x0 + 10, y0 + 10, x1 + 10, y1 + 10, vec3(0.0f), 0.6f);             // เงา
    c.Rect(x0, y0, x1, y1, vec3(0.30f, 0.03f, 0.04f));                           // ตัวหน้าต่าง
    c.Rect(x0, y0, x1, y0 + 40, red);                                            // title bar
    c.Text(x0 + 14, y0 + 10, "FATAL ERROR", 3, white);
    c.Rect(x1 - 40, y0 + 6, x1 - 8, y0 + 34, vec3(0.6f, 0.05f, 0.05f));
    c.Text(x1 - 32, y0 + 10, "X", 3, white);
    c.Frame(x0, y0, x1, y1, 3, vec3(1.0f, 0.3f, 0.3f));

    // ไอคอนวงกลมกากบาท
    c.Disc(x0 + 80.0f, y0 + 115.0f, 46.0f, red);
    c.Line(x0 + 58.0f, y0 + 93.0f, x0 + 102.0f, y0 + 137.0f, 12.0f, white);
    c.Line(x0 + 102.0f, y0 + 93.0f, x0 + 58.0f, y0 + 137.0f, 12.0f, white);

    c.Text(x0 + 160, y0 + 66, "ERROR", 9, white);
    c.Text(x0 + 160, y0 + 146, "SEGMENTATION FAULT", 3, vec3(1.0f, 0.75f, 0.7f));
    c.Text(x0 + 160, y0 + 172, "(CORE DUMPED)", 3, vec3(1.0f, 0.75f, 0.7f));

    // ปุ่ม
    c.Rect(x0 + 160, y0 + 212, x0 + 330, y0 + 252, vec3(0.55f, 0.08f, 0.08f));
    c.Frame(x0 + 160, y0 + 212, x0 + 330, y0 + 252, 3, white);
    c.TextCentered(x0 + 245, y0 + 222, "RETRY", 3, white);
    c.Rect(x0 + 360, y0 + 212, x0 + 560, y0 + 252, vec3(0.55f, 0.08f, 0.08f));
    c.Frame(x0 + 360, y0 + 212, x0 + 560, y0 + 252, 3, white);
    c.TextCentered(x0 + 460, y0 + 222, "GIVE UP", 3, white);

    c.Text(14, 402, "C:\\FUTURE_ME>", 3, green);
    gPromptEndX = 14 + Canvas::TextWidth("C:\\FUTURE_ME>", 3) + 10;
    gPromptY = 402;

    // scanline ของจอ
    for (int y = 0; y < c.h; y += 3)
        for (int x = 0; x < c.w; ++x) c.Set(x, y, vec3(0.0f), 0.18f);
    return c.Upload(false);
}

static GLuint MakeCalendarTexture()      // ปฏิทินปี 2086 ที่ถูกขีดฆ่าทุกวัน
{
    Canvas c(256, 340, vec3(0.92f, 0.90f, 0.84f));
    const vec3 red(0.80f, 0.10f, 0.10f), ink(0.15f, 0.15f, 0.18f);
    c.Rect(0, 0, 256, 78, red);
    c.TextCentered(128, 12, "OCTOBER", 4, vec3(1.0f));
    c.TextCentered(128, 48, "2086", 3, vec3(1.0f));
    for (int i = 0; i < 8; ++i) c.Disc(24.0f + i * 30.0f, 6.0f, 4.0f, vec3(0.2f));   // ห่วงปฏิทิน
    const char* days = "SMTWTFS";
    for (int d = 0; d < 7; ++d) c.Text(14 + d * 34, 90, std::string(1, days[d]), 2, ink);
    for (int day = 1; day <= 31; ++day)
    {
        int slot = day + 3;                       // วันที่ 1 เริ่มวันพฤหัส
        int col = slot % 7, row = slot / 7;
        int x = 10 + col * 34, y = 114 + row * 42;
        c.Text(x, y, std::to_string(day), 2, ink);
        c.Line((float)x - 2, (float)y - 4, (float)x + 24, (float)y + 22, 3.0f, red, 0.85f);
        c.Line((float)x + 24, (float)y - 4, (float)x - 2, (float)y + 22, 3.0f, red, 0.85f);
    }
    return c.Upload(false);
}

static GLuint MakePosterTexture()        // โปสเตอร์ธีม FUTURE ME
{
    Canvas c(240, 340, vec3(0.06f, 0.07f, 0.12f));
    const vec3 yellow(1.0f, 0.82f, 0.25f);
    c.Frame(8, 8, 232, 332, 3, yellow);
    c.TextCentered(120, 30, "FUTURE", 6, yellow);
    c.TextCentered(120, 84, "ME", 6, yellow);
    // หัวกะโหลกเล็กๆ
    c.Disc(120.0f, 182.0f, 36.0f, vec3(0.9f, 0.88f, 0.8f));
    c.Rect(100, 200, 141, 232, vec3(0.9f, 0.88f, 0.8f));
    c.Disc(106.0f, 182.0f, 10.0f, vec3(0.06f, 0.07f, 0.12f));
    c.Disc(134.0f, 182.0f, 10.0f, vec3(0.06f, 0.07f, 0.12f));
    c.Rect(117, 198, 123, 208, vec3(0.06f, 0.07f, 0.12f));
    for (int i = 0; i < 4; ++i) c.Rect(104 + i * 9, 222, 106 + i * 9, 232, vec3(0.06f, 0.07f, 0.12f));
    c.TextCentered(120, 256, "WILL FIX IT", 2, vec3(0.9f));
    c.TextCentered(120, 278, "TOMORROW.", 2, vec3(0.9f));
    c.TextCentered(120, 304, "(PROBABLY)", 2, vec3(0.6f));
    return c.Upload(false);
}

static GLuint MakeStickyTexture(const vec3& paper, const std::vector<std::string>& lines, int scale)
{
    Canvas c(192, 192, paper);
    for (int y = 0; y < c.h; ++y)
        for (int x = 0; x < c.w; ++x)
            c.Set(x, y, paper * (0.92f + 0.08f * Hash2(x, y, 4)));
    c.Rect(0, 0, 192, 20, paper * 0.85f);
    int y = 40;
    for (const std::string& s : lines)
    {
        c.TextCentered(96, y, s, scale, vec3(0.12f, 0.12f, 0.25f));
        y += 9 * scale + 6;
    }
    return c.Upload(false);
}

static GLuint MakeClockTexture()         // หน้าปัดนาฬิกา (นอกวงกลมโปร่งใส)
{
    Canvas c(256, 256, vec3(0.0f), 0.0f);
    c.Disc(128.0f, 128.0f, 126.0f, vec3(0.93f, 0.91f, 0.85f));
    for (int i = 0; i < 60; ++i)
    {
        float a = i / 60.0f * 2.0f * PI;
        float r0 = (i % 5 == 0) ? 100.0f : 110.0f;
        c.Line(128 + std::sin(a) * r0, 128 - std::cos(a) * r0, 128 + std::sin(a) * 118, 128 - std::cos(a) * 118,
               (i % 5 == 0) ? 5.0f : 2.0f, vec3(0.1f));
    }
    c.TextCentered(128, 32, "12", 3, vec3(0.1f));
    c.TextCentered(206, 118, "3", 3, vec3(0.1f));
    c.TextCentered(128, 204, "6", 3, vec3(0.1f));
    c.TextCentered(50, 118, "9", 3, vec3(0.1f));
    return c.Upload(false);
}

static GLuint MakeCanTexture()           // กระป๋องเครื่องดื่มชูกำลัง
{
    Canvas c(256, 128, vec3(0.08f, 0.32f, 0.12f));
    c.Rect(0, 0, 256, 14, vec3(0.75f));
    c.Rect(0, 114, 256, 128, vec3(0.75f));
    c.Rect(0, 40, 256, 88, vec3(0.05f, 0.05f, 0.05f));
    c.Text(10, 50, "ENERGY", 4, vec3(0.6f, 1.0f, 0.3f));
    c.Text(150, 58, "MAX", 3, vec3(1.0f));
    return c.Upload();
}

static GLuint MakeWhiteTexture()
{
    Canvas c(2, 2, vec3(1.0f));
    return c.Upload();
}

// ------------------------------------------------------------------------------------
//  Scene description : Material + DrawItem
// ------------------------------------------------------------------------------------
struct Material
{
    vec3   albedo = vec3(1.0f);
    GLuint tex = 0;
    vec2   uvScale = vec2(1.0f);
    float  spec = 0.1f;
    float  shininess = 16.0f;
    vec3   emissive = vec3(0.0f);
    float  emissiveTex = 0.0f;
    float  alpha = 1.0f;
    bool   doubleSided = false;
};

static Material Mat(const vec3& albedo, float spec = 0.1f, float shininess = 16.0f, GLuint tex = 0, vec2 uvScale = vec2(1.0f))
{
    Material m;
    m.albedo = albedo; m.spec = spec; m.shininess = shininess; m.tex = tex; m.uvScale = uvScale;
    return m;
}

enum ItemTag { TAG_NONE = 0, TAG_CURSOR = 1 };

struct DrawItem
{
    Mesh*    mesh;
    mat4     model;
    Material mat;
    bool     castShadow;
    bool     transparent;
    int      tag;
};

std::vector<DrawItem> gItems;
std::vector<Mesh*>    gMeshes;
Mesh* gCube = nullptr;
Mesh* gSphere = nullptr;
Mesh* gQuad = nullptr;

static Mesh* Keep(Mesh* m) { gMeshes.push_back(m); return m; }

static void Add(Mesh* m, const mat4& M, const Material& mat, bool shadow = true, int tag = TAG_NONE)
{
    gItems.push_back({ m, M, mat, shadow, mat.alpha < 0.999f, tag });
}
static void AddBuilt(const MeshBuilder& mb, const Material& mat, bool shadow = true)
{
    if (mb.idx.empty()) return;
    Add(Keep(mb.Build()), mat4(1.0f), mat, shadow);
}

// ข้อมูลตำแหน่งสำคัญที่ใช้ต่อ (แสงจากจอ, ใยแมงมุม)
struct SceneInfo
{
    vec3 screenCentre, screenNormal, pcLed;
    vec3 monitorTopRight, monitorTopLeft;
    vec3 skullCentre, rShoulder, rElbow, lElbow, rKnee, lKnee, armrestR, mug;
} gScene;

// ------------------------------------------------------------------------------------
//  ห้อง
// ------------------------------------------------------------------------------------
const float ROOM_X0 = -2.0f, ROOM_X1 = 2.4f, ROOM_Z0 = -1.7f, ROOM_Z1 = 3.0f, ROOM_H = 2.7f;
const float WIN_Z0 = -0.35f, WIN_Z1 = 1.05f, WIN_Y0 = 1.0f, WIN_Y1 = 2.2f;

static void BuildRoom()
{
    const float cx = (ROOM_X0 + ROOM_X1) * 0.5f, cz = (ROOM_Z0 + ROOM_Z1) * 0.5f;
    const float sx = ROOM_X1 - ROOM_X0 + 0.2f, sz = ROOM_Z1 - ROOM_Z0 + 0.2f;

    // พื้น
    Add(gCube, T(vec3(cx, -0.05f, cz)) * S(vec3(sx, 0.1f, sz)), Mat(vec3(1.0f), 0.25f, 40.0f, gTex.floor, vec2(5.0f)));

    // ผนัง
    Material wallM = Mat(vec3(1.0f), 0.03f, 8.0f, gTex.wall, vec2(3.5f, 2.2f));
    Add(gCube, T(vec3(cx, ROOM_H * 0.5f, ROOM_Z0 - 0.05f)) * S(vec3(sx, ROOM_H, 0.1f)), wallM);   // หลัง
    Add(gCube, T(vec3(ROOM_X0 - 0.05f, ROOM_H * 0.5f, cz)) * S(vec3(0.1f, ROOM_H, sz)), wallM);   // ซ้าย
    Add(gCube, T(vec3(cx, ROOM_H * 0.5f, ROOM_Z1 + 0.05f)) * S(vec3(sx, ROOM_H, 0.1f)), wallM);   // หน้า
    Add(gCube, T(vec3(cx, ROOM_H + 0.05f, cz)) * S(vec3(sx, 0.1f, sz)), Mat(vec3(0.55f), 0.0f, 4.0f)); // เพดาน

    // ผนังขวามีช่องหน้าต่าง (แสงจันทร์ลอดเข้ามา -> เงาเป็นรูปกรอบหน้าต่าง)
    const float wx = ROOM_X1 + 0.05f;
    auto wallPiece = [&](float y0, float y1, float z0, float z1)
    {
        Add(gCube, T(vec3(wx, (y0 + y1) * 0.5f, (z0 + z1) * 0.5f)) * S(vec3(0.1f, y1 - y0, z1 - z0)), wallM);
    };
    wallPiece(0.0f, WIN_Y0, ROOM_Z0 - 0.1f, ROOM_Z1 + 0.1f);
    wallPiece(WIN_Y1, ROOM_H, ROOM_Z0 - 0.1f, ROOM_Z1 + 0.1f);
    wallPiece(WIN_Y0, WIN_Y1, ROOM_Z0 - 0.1f, WIN_Z0);
    wallPiece(WIN_Y0, WIN_Y1, WIN_Z1, ROOM_Z1 + 0.1f);

    MeshBuilder frame;
    const float fy = (WIN_Y0 + WIN_Y1) * 0.5f, fz = (WIN_Z0 + WIN_Z1) * 0.5f;
    const float wh = WIN_Y1 - WIN_Y0, ww = WIN_Z1 - WIN_Z0;
    frame.Box(T(vec3(wx, fy, fz)) * S(vec3(0.12f, wh, 0.05f)));
    frame.Box(T(vec3(wx, fy, fz)) * S(vec3(0.12f, 0.05f, ww)));
    frame.Box(T(vec3(wx, WIN_Y0 + 0.03f, fz)) * S(vec3(0.14f, 0.06f, ww)));
    frame.Box(T(vec3(wx, WIN_Y1 - 0.03f, fz)) * S(vec3(0.14f, 0.06f, ww)));
    frame.Box(T(vec3(wx, fy, WIN_Z0 + 0.03f)) * S(vec3(0.14f, wh, 0.06f)));
    frame.Box(T(vec3(wx, fy, WIN_Z1 - 0.03f)) * S(vec3(0.14f, wh, 0.06f)));
    AddBuilt(frame, Mat(vec3(0.85f, 0.85f, 0.80f), 0.2f, 32.0f));

    // บัวพื้น
    MeshBuilder skirt;
    skirt.Box(T(vec3(cx, 0.045f, ROOM_Z0 + 0.01f)) * S(vec3(sx, 0.09f, 0.02f)));
    skirt.Box(T(vec3(ROOM_X0 + 0.01f, 0.045f, cz)) * S(vec3(0.02f, 0.09f, sz)));
    AddBuilt(skirt, Mat(vec3(0.25f, 0.17f, 0.12f), 0.2f, 24.0f));

    // โปสเตอร์ "FUTURE ME" บนผนังหลัง
    const vec3 posterPos(-1.30f, 1.50f, ROOM_Z0);
    Add(gCube, T(posterPos + vec3(0.0f, 0.0f, 0.008f)) * S(vec3(0.60f, 0.83f, 0.016f)), Mat(vec3(0.05f), 0.5f, 64.0f));
    Add(gQuad, T(posterPos + vec3(0.0f, 0.0f, 0.0175f)) * S(vec3(0.55f, 0.78f, 1.0f)), Mat(vec3(1.0f), 0.35f, 64.0f, gTex.poster));

    // ปฏิทินบนผนังซ้าย (ปี 2086 ถูกขีดฆ่าทุกวัน)
    mat4 calM = T(vec3(ROOM_X0 + 0.006f, 1.45f, -0.80f)) * R(90.0f, Y_AXIS) * R(-4.0f, Z_AXIS);
    Add(gQuad, calM * S(vec3(0.36f, 0.48f, 1.0f)), Mat(vec3(1.0f), 0.05f, 8.0f, gTex.calendar));
    Add(gSphere, calM * T(vec3(0.0f, 0.26f, 0.0f)) * S(0.008f), Mat(vec3(0.6f), 0.8f, 64.0f));

    // นาฬิกาแขวน (เข็มชี้ 23:59 = เวลาส่งงาน!)
    const vec3 cc(-0.55f, 1.86f, ROOM_Z0 + 0.03f);
    Material face = Mat(vec3(1.0f), 0.2f, 32.0f, gTex.clock);
    Add(gQuad, T(cc) * S(vec3(0.30f, 0.30f, 1.0f)), face);
    MeshBuilder rim;
    std::vector<vec3> ring; std::vector<float> rr;
    for (int i = 0; i <= 48; ++i)
    {
        float a = 2.0f * PI * i / 48.0f;
        ring.push_back(cc + vec3(std::cos(a) * 0.152f, std::sin(a) * 0.152f, 0.0f));
        rr.push_back(0.014f);
    }
    rim.Tube(ring, rr, 10, false);
    AddBuilt(rim, Mat(vec3(0.08f), 0.6f, 64.0f));
    auto hand = [&](float angleDeg, float len, float wdt, const vec3& col, float z)
    {
        Add(gCube, T(cc + vec3(0.0f, 0.0f, z)) * R(-angleDeg, Z_AXIS) * T(vec3(0.0f, len * 0.5f - 0.02f, 0.0f)) * S(vec3(wdt, len, 0.004f)),
            Mat(col, 0.3f, 32.0f), false);
    };
    hand((11.0f + 59.0f / 60.0f) * 30.0f, 0.085f, 0.010f, vec3(0.05f), 0.006f);   // เข็มชั่วโมง
    hand(59.0f * 6.0f, 0.120f, 0.007f, vec3(0.05f), 0.010f);                        // เข็มนาที
    hand(45.0f * 6.0f, 0.125f, 0.003f, vec3(0.8f, 0.1f, 0.1f), 0.014f);             // เข็มวินาที
    Add(gSphere, T(cc + vec3(0.0f, 0.0f, 0.016f)) * S(0.008f), Mat(vec3(0.1f)), false);
}

// ------------------------------------------------------------------------------------
//  โต๊ะทำงาน
// ------------------------------------------------------------------------------------
static void BuildDesk()
{
    Material wood = Mat(vec3(1.0f), 0.35f, 48.0f, gTex.wood);
    Add(gCube, T(vec3(-0.05f, 0.73f, -1.24f)) * S(vec3(1.70f, 0.04f, 0.80f)), wood);                // ท็อปโต๊ะ

    // ตู้ลิ้นชักด้านซ้าย
    Add(gCube, T(vec3(-0.70f, 0.355f, -1.24f)) * S(vec3(0.40f, 0.71f, 0.76f)), Mat(vec3(0.8f), 0.25f, 32.0f, gTex.wood, vec2(0.6f, 1.0f)));
    MeshBuilder drawers, handles;
    const float dy[3] = { 0.585f, 0.37f, 0.14f }, dh[3] = { 0.20f, 0.20f, 0.22f };
    for (int i = 0; i < 3; ++i)
    {
        drawers.Box(T(vec3(-0.70f, dy[i], -0.855f)) * S(vec3(0.36f, dh[i], 0.016f)));
        handles.Box(T(vec3(-0.70f, dy[i] + 0.04f, -0.842f)) * S(vec3(0.12f, 0.016f, 0.012f)));
    }
    AddBuilt(drawers, Mat(vec3(1.05f, 1.0f, 0.95f), 0.3f, 32.0f, gTex.wood, vec2(0.5f, 0.3f)));
    AddBuilt(handles, Mat(vec3(0.7f, 0.7f, 0.72f), 0.9f, 96.0f));

    // ขาโต๊ะเหล็กด้านขวา + แผ่นบังด้านหลัง
    MeshBuilder legs;
    legs.Box(T(vec3(0.76f, 0.355f, -1.58f)) * S(vec3(0.04f, 0.71f, 0.04f)));
    legs.Box(T(vec3(0.76f, 0.355f, -0.90f)) * S(vec3(0.04f, 0.71f, 0.04f)));
    legs.Box(T(vec3(0.76f, 0.08f, -1.24f)) * S(vec3(0.035f, 0.035f, 0.66f)));
    AddBuilt(legs, Mat(vec3(0.12f, 0.12f, 0.13f), 0.7f, 64.0f));
    Add(gCube, T(vec3(0.12f, 0.48f, -1.60f)) * S(vec3(1.22f, 0.44f, 0.02f)), Mat(vec3(0.7f), 0.2f, 16.0f, gTex.wood, vec2(1.0f, 0.4f)));
}

// ------------------------------------------------------------------------------------
//  คอมพิวเตอร์ : จอ + คีย์บอร์ด + เมาส์ + เคส
// ------------------------------------------------------------------------------------
static void BuildComputer()
{
    Material plastic = Mat(vec3(0.07f, 0.07f, 0.08f), 0.55f, 64.0f);

    // ---- จอมอนิเตอร์ (หันเข้าหากล้องเล็กน้อย) ----
    const mat4 Mmon = T(vec3(-0.08f, 0.75f, -1.36f)) * R(18.0f, Y_AXIS);
    MeshBuilder mon;
    mon.Box(Mmon * T(vec3(0.0f, 0.01f, 0.0f)) * S(vec3(0.28f, 0.02f, 0.20f)));      // ฐาน
    mon.Box(Mmon * T(vec3(0.0f, 0.19f, -0.045f)) * S(vec3(0.06f, 0.36f, 0.03f)));   // คอ
    mon.Box(Mmon * T(vec3(0.0f, 0.405f, 0.0f)) * S(vec3(0.78f, 0.46f, 0.035f)));    // กรอบจอ
    mon.Box(Mmon * T(vec3(0.0f, 0.38f, -0.035f)) * S(vec3(0.42f, 0.28f, 0.05f)));   // ด้านหลังจอ
    AddBuilt(mon, plastic);

    const float SW = 0.74f, SH = 0.416f;                                              // 16:9
    const mat4 Mscr = Mmon * T(vec3(0.0f, 0.405f, 0.0182f));
    Material screen = Mat(vec3(1.0f), 0.9f, 160.0f, gTex.screen);
    screen.emissiveTex = 3.2f;
    Add(gQuad, Mscr * S(vec3(SW, SH, 1.0f)), screen, false);

    // cursor กะพริบ (ต่อท้าย prompt บรรทัดล่าง)
    float cxl = (gPromptEndX + 7.5f) / SCREEN_TEX_W - 0.5f;
    float cyl = 0.5f - (gPromptY + 10.5f) / SCREEN_TEX_H;
    Material cursor = Mat(vec3(0.0f));
    cursor.emissive = vec3(0.5f, 1.0f, 0.6f);
    Add(gQuad, Mscr * T(vec3(0.0f, 0.0f, 0.0006f)) * S(vec3(SW, SH, 1.0f)) * T(vec3(cxl, cyl, 0.0f)) * S(vec3(15.0f / SCREEN_TEX_W, 21.0f / SCREEN_TEX_H, 1.0f)),
        cursor, false, TAG_CURSOR);

    // ไฟ power ของจอ
    Material ledBlue = Mat(vec3(0.1f));
    ledBlue.emissive = vec3(0.3f, 0.6f, 1.0f);
    Add(gSphere, Mmon * T(vec3(0.35f, 0.19f, 0.018f)) * S(0.004f), ledBlue, false);

    gScene.screenCentre    = xform(Mmon, vec3(0.0f, 0.405f, 0.02f));
    gScene.screenNormal    = glm::normalize(xdir(Mmon, Z_AXIS));
    gScene.monitorTopRight = xform(Mmon, vec3(0.39f, 0.635f, -0.01f));
    gScene.monitorTopLeft  = xform(Mmon, vec3(-0.39f, 0.635f, -0.01f));

    // โพสต์อิทติดขอบจอ
    Material note1 = Mat(vec3(1.0f), 0.05f, 8.0f, gTex.sticky1);
    note1.doubleSided = true;
    Add(gQuad, Mmon * T(vec3(-0.27f, 0.645f, 0.0185f)) * R(-7.0f, Z_AXIS) * S(vec3(0.10f, 0.10f, 1.0f)), note1);
    Material note2 = Mat(vec3(1.0f), 0.05f, 8.0f, gTex.sticky2);
    note2.doubleSided = true;
    Add(gQuad, Mmon * T(vec3(0.335f, 0.64f, 0.0185f)) * R(9.0f, Z_AXIS) * S(vec3(0.10f, 0.10f, 1.0f)), note2);

    // ---- คีย์บอร์ด ----
    const mat4 Mkb = T(vec3(-0.10f, 0.75f, -1.00f)) * R(3.0f, Y_AXIS);
    MeshBuilder kbBase, keys;
    kbBase.Box(Mkb * T(vec3(0.0f, 0.011f, 0.0f)) * S(vec3(0.44f, 0.022f, 0.15f)));
    for (int row = 0; row < 5; ++row)
    {
        float z = -0.052f + row * 0.026f;
        if (row < 4)
        {
            for (int k = 0; k < 14; ++k)
            {
                float x = -0.195f + k * 0.030f + (row % 2) * 0.006f;
                if (x > 0.20f) continue;
                keys.Box(Mkb * T(vec3(x, 0.027f, z)) * S(vec3(0.025f, 0.012f, 0.022f)));
            }
        }
        else
        {
            keys.Box(Mkb * T(vec3(-0.17f, 0.027f, z)) * S(vec3(0.05f, 0.012f, 0.022f)));
            keys.Box(Mkb * T(vec3(-0.01f, 0.027f, z)) * S(vec3(0.20f, 0.012f, 0.022f)));    // spacebar
            keys.Box(Mkb * T(vec3(0.16f, 0.027f, z)) * S(vec3(0.07f, 0.012f, 0.022f)));
        }
    }
    AddBuilt(kbBase, Mat(vec3(0.05f), 0.4f, 48.0f));
    AddBuilt(keys, Mat(vec3(0.20f, 0.20f, 0.22f), 0.35f, 48.0f));

    // ---- แผ่นรองเมาส์ + เมาส์ ----
    Add(gCube, T(vec3(0.34f, 0.7515f, -1.03f)) * R(-6.0f, Y_AXIS) * S(vec3(0.25f, 0.003f, 0.21f)), Mat(vec3(0.08f, 0.09f, 0.14f), 0.1f, 8.0f));
    MeshBuilder mouse;
    mouse.Sphere(T(vec3(0.34f, 0.755f, -1.00f)) * R(-10.0f, Y_AXIS) * S(vec3(0.031f, 0.020f, 0.052f)));
    std::vector<vec3> cable = { vec3(0.34f, 0.757f, -1.05f), vec3(0.35f, 0.758f, -1.15f), vec3(0.30f, 0.758f, -1.30f),
                                vec3(0.22f, 0.758f, -1.45f), vec3(0.20f, 0.758f, -1.62f) };
    mouse.Tube(cable, std::vector<float>(cable.size(), 0.0025f), 6, false);
    AddBuilt(mouse, Mat(vec3(0.10f), 0.7f, 64.0f));

    // ---- เคสคอม (วางพื้นข้างโต๊ะ) ----
    const mat4 Mpc = T(vec3(1.06f, 0.0f, -1.28f)) * R(-8.0f, Y_AXIS);
    MeshBuilder pc, vents;
    pc.Box(Mpc * T(vec3(0.0f, 0.24f, 0.0f)) * S(vec3(0.21f, 0.45f, 0.46f)));
    for (int i = 0; i < 4; ++i)
        pc.Box(Mpc * T(vec3((i % 2 ? 0.08f : -0.08f), 0.008f, (i / 2 ? 0.19f : -0.19f))) * S(vec3(0.03f, 0.016f, 0.03f)));
    for (int i = 0; i < 9; ++i)
        vents.Box(Mpc * T(vec3(0.0f, 0.08f + i * 0.022f, 0.231f)) * S(vec3(0.15f, 0.008f, 0.004f)));
    AddBuilt(pc, Mat(vec3(0.10f, 0.10f, 0.11f), 0.5f, 48.0f));
    AddBuilt(vents, Mat(vec3(0.02f), 0.1f, 8.0f));
    MeshBuilder pcLeds;
    pcLeds.Cylinder(xform(Mpc, vec3(0.0f, 0.39f, 0.230f)), xform(Mpc, vec3(0.0f, 0.39f, 0.238f)), 0.012f, 0.012f, 16);
    pcLeds.Box(Mpc * T(vec3(-0.095f, 0.24f, 0.232f)) * S(vec3(0.006f, 0.38f, 0.004f)));
    Material ledM = Mat(vec3(0.1f));
    ledM.emissive = vec3(0.25f, 0.55f, 1.0f);
    AddBuilt(pcLeds, ledM, false);
    gScene.pcLed = xform(Mpc, vec3(-0.05f, 0.30f, 0.33f));

    // สายไฟจากจอลงไปที่เคส
    MeshBuilder cables;
    std::vector<vec3> c1 = { xform(Mmon, vec3(0.0f, 0.30f, -0.06f)), xform(Mmon, vec3(0.05f, 0.10f, -0.12f)),
                             vec3(0.30f, 0.76f, -1.58f), vec3(0.62f, 0.74f, -1.66f), vec3(0.85f, 0.45f, -1.66f),
                             vec3(0.98f, 0.30f, -1.55f), xform(Mpc, vec3(0.0f, 0.35f, -0.23f)) };
    cables.Tube(c1, std::vector<float>(c1.size(), 0.004f), 6, false);
    AddBuilt(cables, Mat(vec3(0.03f), 0.4f, 32.0f));
}

// กล่องขอบมน : กล่อง 3 ชิ้นไขว้กัน + ทรงกระบอก 12 ขอบ + ทรงกลม 8 มุม
static void RoundedBox(MeshBuilder& mb, const mat4& M, const vec3& size, float r)
{
    const vec3 h = size * 0.5f - vec3(r);
    mb.Box(M * S(vec3(size.x, h.y * 2.0f, h.z * 2.0f)));
    mb.Box(M * S(vec3(h.x * 2.0f, size.y, h.z * 2.0f)));
    mb.Box(M * S(vec3(h.x * 2.0f, h.y * 2.0f, size.z)));
    for (int a = -1; a <= 1; a += 2){
        for (int b = -1; b <= 1; b += 2)
        {
            mb.Cylinder(xform(M, vec3(-h.x, a * h.y, b * h.z)), xform(M, vec3(h.x, a * h.y, b * h.z)), r, r, 10, false);
            mb.Cylinder(xform(M, vec3(a * h.x, -h.y, b * h.z)), xform(M, vec3(a * h.x, h.y, b * h.z)), r, r, 10, false);
            mb.Cylinder(xform(M, vec3(a * h.x, b * h.y, -h.z)), xform(M, vec3(a * h.x, b * h.y, h.z)), r, r, 10, false);
            for (int c = -1; c <= 1; c += 2)
                mb.Sphere(M * T(vec3(a * h.x, b * h.y, c * h.z)) * S(r), 10, 8);
        }
    }
}

// ------------------------------------------------------------------------------------
//  เก้าอี้สำนักงาน
// ------------------------------------------------------------------------------------
static void BuildChair()
{
    const mat4 Mch = T(vec3(-0.05f, 0.0f, -0.46f)) * R(-15.0f, Y_AXIS);
    MeshBuilder metal, fabric, plastic;

    const vec3 hub(0.0f, 0.10f, 0.0f);
    for (int k = 0; k < 5; ++k)
    {
        float a = glm::radians(18.0f + k * 72.0f);
        vec3 dir(std::sin(a), 0.0f, std::cos(a));
        vec3 end = hub + dir * 0.30f + vec3(0.0f, -0.025f, 0.0f);
        metal.Tube({ xform(Mch, hub), xform(Mch, end) }, { 0.026f, 0.018f }, 8, true);
        plastic.Sphere(Mch * T(end + vec3(0.0f, -0.040f, 0.0f)) * S(vec3(0.028f, 0.030f, 0.020f)), 12, 8);
    }
    metal.Cylinder(xform(Mch, vec3(0.0f, 0.07f, 0.0f)), xform(Mch, vec3(0.0f, 0.27f, 0.0f)), 0.040f, 0.036f, 16);
    metal.Cylinder(xform(Mch, vec3(0.0f, 0.27f, 0.0f)), xform(Mch, vec3(0.0f, 0.41f, 0.0f)), 0.022f, 0.022f, 12);
    metal.Box(Mch * T(vec3(0.0f, 0.39f, 0.02f)) * S(vec3(0.22f, 0.04f, 0.24f)));

    RoundedBox(fabric, Mch * T(vec3(0.0f, 0.44f, 0.0f)), vec3(0.50f, 0.08f, 0.48f), 0.03f);                                       // เบาะ
    RoundedBox(fabric, Mch * T(vec3(0.0f, 0.50f, 0.255f)) * R(12.0f, X_AXIS) * T(vec3(0.0f, 0.30f, 0.0f)), vec3(0.46f, 0.44f, 0.06f), 0.028f);   // พนักพิง
    metal.Box(Mch * T(vec3(0.0f, 0.43f, 0.26f)) * R(12.0f, X_AXIS) * T(vec3(0.0f, 0.06f, 0.0f)) * S(vec3(0.07f, 0.20f, 0.025f)));

    for (int s = -1; s <= 1; s += 2)                                                                              // ที่วางแขน
    {
        metal.Box(Mch * T(vec3(s * 0.265f, 0.55f, 0.05f)) * S(vec3(0.03f, 0.20f, 0.04f)));
        RoundedBox(plastic, Mch * T(vec3(s * 0.265f, 0.66f, 0.0f)), vec3(0.07f, 0.03f, 0.27f), 0.012f);
    }
    gScene.armrestR = xform(Mch, vec3(0.265f, 0.675f, -0.12f));

    AddBuilt(metal, Mat(vec3(0.15f, 0.15f, 0.16f), 0.8f, 96.0f));
    AddBuilt(plastic, Mat(vec3(0.05f), 0.4f, 32.0f));
    AddBuilt(fabric, Mat(vec3(1.0f), 0.05f, 8.0f, gTex.fabric, vec2(3.0f)));
}

// ------------------------------------------------------------------------------------
//  โครงกระดูก "FUTURE ME"
// ------------------------------------------------------------------------------------
static void BuildSkeleton(){

    // โหลดโมเดลกระดูกจากไฟล์ OBJ ในโฟลเดอร์ Models
    // เปลี่ยนชื่อ "skeleton.obj" ตรงนี้ได้ หากไฟล์ของคุณใช้ชื่ออื่น
    std::filesystem::path modelPath;
    std::filesystem::path folder = std::filesystem::current_path();
    for (int i = 0; i < 6; ++i)
    {
        const std::filesystem::path candidate = folder / "Models" / "skeleton.obj";
        if (std::filesystem::exists(candidate))
        {
            modelPath = candidate;
            break;
        }
        if (folder == folder.root_path()) break;
        folder = folder.parent_path();
    }
    if (modelPath.empty()) modelPath = std::filesystem::current_path() / "Models" / "skeleton.obj";

    Mesh* skeleton = new Mesh();
    if (!skeleton->CreateMeshFromOBJ(modelPath.string(), true))
    {
        std::cerr << "[Skeleton] โหลดโมเดลไม่สำเร็จ: " << modelPath.string() << "\n"
                  << "กรุณาวางไฟล์ OBJ ไว้ในโฟลเดอร์ Models และตรวจสอบชื่อไฟล์\n";
        delete skeleton;
        return;
    }

    // ตำแหน่งข้างโต๊ะคอม: ปรับค่า scale/ตำแหน่งได้ตามขนาดโมเดลที่ export จาก Blender
    const vec3 skeletonPosition(-0.05f, -0.12f, -0.50f);
    const float skeletonScale = 0.04f;
    const mat4 skeletonModel = T(skeletonPosition) * R(-185.0f, Y_AXIS) * S(skeletonScale);
    Add(Keep(skeleton), skeletonModel, Mat(vec3(0.84f, 0.80f, 0.68f), 0.30f, 28.0f));

    // จุดอ้างอิงของฉากที่ส่วนใยแมงมุม/เอฟเฟกต์อื่นใช้อยู่
    // ตั้งให้สอดคล้องกับตำแหน่งโดยประมาณของโมเดลยืน
    gScene.skullCentre = skeletonPosition + vec3(0.0f, 0.78f * skeletonScale, 0.0f);
    gScene.rShoulder   = skeletonPosition + vec3(0.12f * skeletonScale, 0.58f * skeletonScale, 0.0f);
    gScene.lElbow      = skeletonPosition + vec3(-0.18f * skeletonScale, 0.42f * skeletonScale, 0.0f);
    gScene.rElbow      = skeletonPosition + vec3(0.18f * skeletonScale, 0.42f * skeletonScale, 0.0f);
    gScene.rKnee       = skeletonPosition + vec3(0.08f * skeletonScale, 0.22f * skeletonScale, 0.0f);
    gScene.lKnee       = skeletonPosition + vec3(-0.08f * skeletonScale, 0.22f * skeletonScale, 0.0f);
}

// ------------------------------------------------------------------------------------
//  ของบนโต๊ะ : แก้วกาแฟขึ้นรา, กระป๋องเครื่องดื่ม, ต้นไม้ตาย, กองกระดาษ, เป็ดยาง (rubber duck debugging)
// ------------------------------------------------------------------------------------
static void BuildProps()
{
    // ---- แก้วกาแฟ ----
    const vec3 mugPos(0.55f, 0.75f, -1.16f);
    MeshBuilder mug;
    mug.Lathe({ {0.0f, 0.0f}, {0.036f, 0.0f}, {0.039f, 0.004f}, {0.040f, 0.020f}, {0.040f, 0.090f}, {0.038f, 0.095f},
                {0.035f, 0.093f}, {0.034f, 0.012f}, {0.0f, 0.012f} }, T(mugPos), 28);
    std::vector<vec3> handle;
    for (int i = 0; i <= 12; ++i)
    {
        float a = glm::radians(-80.0f + 160.0f * i / 12.0f);
        handle.push_back(mugPos + vec3(0.038f + 0.024f * std::cos(a), 0.05f + 0.026f * std::sin(a), 0.0f));
    }
    mug.Tube(handle, std::vector<float>(handle.size(), 0.0055f), 8, false);
    AddBuilt(mug, Mat(vec3(0.72f, 0.12f, 0.12f), 0.6f, 64.0f));
    MeshBuilder mold;
    mold.Lathe({ {0.0345f, 0.0f}, {0.0f, 0.0f} }, T(mugPos + vec3(0.0f, 0.062f, 0.0f)), 24);
    AddBuilt(mold, Mat(vec3(0.20f, 0.24f, 0.12f), 0.05f, 8.0f));
    gScene.mug = mugPos + vec3(0.0f, 0.095f, 0.0f);

    // ---- กระป๋องเครื่องดื่มชูกำลัง ----
    MeshBuilder cans;
    const std::vector<vec2> canProf = { {0.0f, 0.0f}, {0.026f, 0.0f}, {0.033f, 0.008f}, {0.033f, 0.112f}, {0.027f, 0.122f}, {0.0f, 0.122f} };
    cans.Lathe(canProf, T(vec3(0.64f, 0.75f, -1.50f)) * R(30.0f, Y_AXIS), 24);
    cans.Lathe(canProf, T(vec3(0.715f, 0.75f, -1.42f)) * R(-60.0f, Y_AXIS), 24);
    cans.Lathe(canProf, T(vec3(0.63f, 0.75f, -1.37f)) * R(100.0f, Y_AXIS), 24);
    cans.Lathe(canProf, T(vec3(0.40f, 0.783f, -1.22f)) * R(70.0f, Y_AXIS) * R(90.0f, Z_AXIS) * T(vec3(0.0f, -0.061f, 0.0f)), 24);
    cans.Lathe(canProf, T(vec3(0.55f, 0.033f, -0.05f)) * R(20.0f, Y_AXIS) * R(90.0f, Z_AXIS) * T(vec3(0.0f, -0.061f, 0.0f)), 24);
    cans.Lathe(canProf, T(vec3(0.80f, 0.033f, 0.30f)) * R(-50.0f, Y_AXIS) * R(90.0f, Z_AXIS) * T(vec3(0.0f, -0.061f, 0.0f)), 24);
    cans.Lathe(canProf, T(vec3(0.95f, 0.0f, -0.25f)) * R(150.0f, Y_AXIS), 24);
    AddBuilt(cans, Mat(vec3(1.0f), 0.8f, 96.0f, gTex.can));

    // ---- ต้นไม้ที่ตายไปนานแล้ว ----
    const vec3 potPos(-0.70f, 0.75f, -1.46f);
    MeshBuilder pot, soil, stems, leaves;
    pot.Lathe({ {0.0f, 0.0f}, {0.050f, 0.0f}, {0.064f, 0.110f}, {0.072f, 0.114f}, {0.072f, 0.130f}, {0.064f, 0.130f}, {0.058f, 0.110f} }, T(potPos), 28);
    soil.Lathe({ {0.059f, 0.0f}, {0.0f, 0.0f} }, T(potPos + vec3(0.0f, 0.108f, 0.0f)), 24);
    for (int k = 0; k < 6; ++k)
    {
        float a = glm::radians(20.0f + k * 61.0f);
        vec3 d(std::cos(a), 0.0f, std::sin(a));
        float h = 0.16f + 0.05f * Hash2(k, 1, 3);
        vec3 b = potPos + vec3(0.0f, 0.11f, 0.0f) + d * 0.012f;
        std::vector<vec3> stem = { b, b + vec3(0.0f, h * 0.5f, 0.0f) + d * 0.01f, b + vec3(0.0f, h, 0.0f) + d * 0.04f,
                                   b + vec3(0.0f, h * 0.95f, 0.0f) + d * 0.08f, b + vec3(0.0f, h * 0.70f, 0.0f) + d * 0.11f,
                                   b + vec3(0.0f, h * 0.45f, 0.0f) + d * 0.12f };
        stems.Tube(stem, { 0.0035f, 0.003f, 0.0026f, 0.0022f, 0.0018f, 0.0015f }, 6, true);
        vec3 tip = stem.back();
        leaves.Sphere(T(tip + vec3(0.0f, -0.015f, 0.0f)) * R(glm::degrees(-a), Y_AXIS) * R(70.0f, Z_AXIS) * S(vec3(0.020f, 0.004f, 0.012f)), 10, 6);
        vec3 mid = stem[3];
        leaves.Sphere(T(mid + vec3(0.0f, -0.01f, 0.0f)) * R(glm::degrees(-a) + 40.0f, Y_AXIS) * R(55.0f, Z_AXIS) * S(vec3(0.018f, 0.004f, 0.011f)), 10, 6);
    }
    for (int k = 0; k < 4; ++k)                                     // ใบไม้แห้งที่ร่วงบนโต๊ะ
        leaves.Sphere(T(potPos + vec3(0.06f + 0.05f * k, 0.002f, 0.08f + 0.03f * Hash2(k, 7, 1))) * R(k * 47.0f, Y_AXIS) *
                      S(vec3(0.018f, 0.002f, 0.011f)), 10, 6);
    AddBuilt(pot, Mat(vec3(0.62f, 0.33f, 0.21f), 0.1f, 12.0f));
    AddBuilt(soil, Mat(vec3(0.16f, 0.11f, 0.08f), 0.02f, 4.0f));
    AddBuilt(stems, Mat(vec3(0.35f, 0.26f, 0.15f), 0.05f, 8.0f));
    AddBuilt(leaves, Mat(vec3(0.50f, 0.36f, 0.16f), 0.05f, 8.0f));

    // ---- กองกระดาษ / กระดาษขยำ ----
    MeshBuilder papers;
    for (int i = 0; i < 7; ++i)
        papers.Box(T(vec3(-0.47f + 0.012f * (Hash2(i, 2, 9) - 0.5f), 0.752f + 0.0028f * i, -1.10f + 0.012f * (Hash2(i, 5, 9) - 0.5f))) *
                   R(14.0f + 9.0f * (Hash2(i, 8, 2) - 0.5f), Y_AXIS) * S(vec3(0.21f, 0.0022f, 0.297f)));
    const vec3 balls[5] = { vec3(0.15f, 0.776f, -1.22f), vec3(-0.40f, 0.026f, 0.05f), vec3(0.32f, 0.026f, 0.38f),
                            vec3(-0.85f, 0.026f, -0.55f), vec3(0.62f, 0.026f, -0.55f) };
    for (int i = 0; i < 5; ++i)
    {
        papers.Sphere(T(balls[i]) * R(i * 37.0f, Y_AXIS) * S(vec3(0.026f, 0.024f, 0.027f)), 10, 8);
        papers.Sphere(T(balls[i] + vec3(0.008f, 0.006f, -0.006f)) * S(0.018f), 8, 6);
    }
    AddBuilt(papers, Mat(vec3(1.0f), 0.05f, 8.0f, gTex.paper));

    // ---- เป็ดยางเพื่อนดีบัก ----
    const mat4 Md = T(vec3(0.42f, 0.75f, -1.52f)) * R(25.0f, Y_AXIS);
    MeshBuilder duck, beak, eyes;
    duck.Sphere(Md * T(vec3(0.0f, 0.036f, 0.0f)) * S(vec3(0.045f, 0.036f, 0.056f)));
    duck.Sphere(Md * T(vec3(0.0f, 0.056f, -0.045f)) * R(-35.0f, X_AXIS) * S(vec3(0.018f, 0.012f, 0.022f)));
    duck.Sphere(Md * T(vec3(0.0f, 0.090f, 0.024f)) * S(0.029f));
    beak.Sphere(Md * T(vec3(0.0f, 0.084f, 0.054f)) * S(vec3(0.016f, 0.007f, 0.016f)));
    for (int s = -1; s <= 1; s += 2) eyes.Sphere(Md * T(vec3(s * 0.013f, 0.098f, 0.047f)) * S(0.0045f), 8, 6);
    AddBuilt(duck, Mat(vec3(1.0f, 0.82f, 0.10f), 0.6f, 64.0f));
    AddBuilt(beak, Mat(vec3(1.0f, 0.45f, 0.05f), 0.4f, 32.0f));
    AddBuilt(eyes, Mat(vec3(0.02f), 0.9f, 128.0f));
}

// ------------------------------------------------------------------------------------
//  ใยแมงมุม (เส้นบางมาก + โปร่งแสง)
// ------------------------------------------------------------------------------------
static void Cobweb(MeshBuilder& mb, const vec3& centre, const std::vector<vec3>& anchors, bool closed, int rings, int seed)
{
    const float r = 0.0010f;
    const size_t n = anchors.size();
    std::vector<float> sagAmt(n);
    auto radial = [&](size_t i, float f) -> vec3
    {
        return lerp3(centre, anchors[i], f) + vec3(0.0f, -sagAmt[i] * 4.0f * f * (1.0f - f), 0.0f);
    };
    for (size_t i = 0; i < n; ++i)
    {
        sagAmt[i] = 0.025f * glm::length(anchors[i] - centre);
        std::vector<vec3> pts;
        for (int k = 0; k <= 6; ++k) pts.push_back(radial(i, k / 6.0f));
        mb.Tube(pts, std::vector<float>(pts.size(), r), 4, false);
    }
    const size_t segs = closed ? n : n - 1;
    for (int k = 1; k <= rings; ++k)
        for (size_t i = 0; i < segs; ++i)
        {
            size_t j = (i + 1) % n;
            float fa = (k + 0.35f * (Hash2(k, (int)i, seed) - 0.5f)) / (rings + 1.0f);
            float fb = (k + 0.35f * (Hash2(k, (int)j, seed) - 0.5f)) / (rings + 1.0f);
            vec3 a = radial(i, fa), b = radial(j, fb);
            vec3 m = lerp3(a, b, 0.5f);
            m = m + (centre - m) * 0.12f + vec3(0.0f, -0.004f, 0.0f);
            mb.Tube({ a, lerp3(a, m, 0.6f), m, lerp3(b, m, 0.6f), b }, std::vector<float>(5, r), 4, false);
        }
}

static void BuildCobwebs()
{
    MeshBuilder web;
    const vec3 sk = gScene.skullCentre;

    // หัวกะโหลก <-> ไหล่ขวา
    {
        vec3 c = lerp3(sk, gScene.rShoulder, 0.5f) + vec3(0.03f, 0.0f, 0.02f);
        Cobweb(web, c, { sk + vec3(0.06f, 0.03f, 0.0f), sk + vec3(0.07f, -0.04f, 0.03f), gScene.rShoulder + vec3(0.0f, 0.02f, 0.0f),
                         lerp3(gScene.rShoulder, gScene.rElbow, 0.3f), gScene.rShoulder + vec3(0.04f, 0.05f, 0.05f) }, true, 7, 1);
    }
    // ข้อศอกขวา <-> ที่วางแขนเก้าอี้
    {
        vec3 c = lerp3(gScene.rElbow, gScene.armrestR, 0.5f);
        Cobweb(web, c, { gScene.rElbow, lerp3(gScene.rElbow, gScene.rShoulder, 0.25f), gScene.armrestR + vec3(0.0f, 0.0f, 0.10f),
                         gScene.armrestR, gScene.armrestR + vec3(0.0f, 0.0f, -0.12f) }, true, 6, 2);
    }
    // มุมห้อง (ผนังซ้าย/ผนังหลัง)
    {
        vec3 c(ROOM_X0 + 0.12f, 1.10f, ROOM_Z0 + 0.12f);
        Cobweb(web, c, { vec3(ROOM_X0, 1.55f, ROOM_Z0 + 0.35f), vec3(ROOM_X0, 1.30f, ROOM_Z0 + 0.55f), vec3(ROOM_X0, 0.80f, ROOM_Z0 + 0.45f),
                         vec3(ROOM_X0 + 0.40f, 0.75f, ROOM_Z0), vec3(ROOM_X0 + 0.55f, 1.25f, ROOM_Z0), vec3(ROOM_X0 + 0.30f, 1.60f, ROOM_Z0) }, false, 9, 3);
    }
    // มุมจอ <-> ผนัง
    {
        vec3 a = gScene.monitorTopRight;
        vec3 c = a + vec3(0.12f, 0.10f, -0.12f);
        Cobweb(web, c, { a, a + vec3(0.0f, -0.20f, 0.0f), vec3(a.x + 0.25f, 0.752f, ROOM_Z0 + 0.10f), vec3(a.x + 0.35f, a.y + 0.05f, ROOM_Z0),
                         vec3(a.x + 0.10f, a.y + 0.30f, ROOM_Z0) }, false, 7, 4);
    }
    {   // แมงมุมตัวเล็กเฝ้าใย
        vec3 c = gScene.monitorTopRight + vec3(0.12f, 0.10f, -0.12f) + vec3(0.0f, -0.012f, 0.0f);
        MeshBuilder spider;
        spider.Sphere(T(c) * S(vec3(0.011f, 0.009f, 0.013f)), 12, 8);
        spider.Sphere(T(c + vec3(0.0f, 0.004f, 0.016f)) * S(0.007f), 10, 6);
        for (int s = -1; s <= 1; s += 2)
            for (int k = 0; k < 4; ++k)
            {
                float a = glm::radians(-50.0f + k * 33.0f);
                vec3 out(s * std::cos(a), 0.0f, std::sin(a));
                spider.Tube({ c, c + out * 0.018f + vec3(0.0f, 0.010f, 0.0f), c + out * 0.032f + vec3(0.0f, -0.006f, 0.0f) },
                            { 0.0016f, 0.0013f, 0.0009f }, 5, false);
            }
        AddBuilt(spider, Mat(vec3(0.03f), 0.6f, 64.0f), false);
    }
    // แก้วกาแฟ <-> กระป๋อง
    {
        vec3 m = gScene.mug;
        vec3 c = m + vec3(0.04f, 0.03f, -0.10f);
        Cobweb(web, c, { m + vec3(0.0f, 0.0f, -0.035f), vec3(0.63f, 0.87f, -1.37f), vec3(0.715f, 0.85f, -1.42f), vec3(0.68f, 0.751f, -1.25f) }, true, 5, 5);
    }
    // หน้าแข้ง <-> ขาเก้าอี้ / พื้น
    {
        vec3 k = gScene.rKnee;
        vec3 c = vec3(k.x + 0.05f, 0.25f, k.z + 0.18f);
        Cobweb(web, c, { lerp3(k, vec3(0.11f, 0.085f, -0.90f), 0.35f), vec3(k.x + 0.02f, 0.0f, k.z + 0.30f),
                         vec3(0.20f, 0.06f, -0.55f), vec3(0.05f, 0.30f, -0.36f) }, true, 6, 6);
    }

    Material webM = Mat(vec3(0.85f, 0.85f, 0.88f), 0.6f, 32.0f);
    webM.alpha = 0.45f;
    webM.emissive = vec3(0.10f);
    webM.doubleSided = true;
    AddBuilt(web, webM, false);
}

// ------------------------------------------------------------------------------------
//  Rendering
// ------------------------------------------------------------------------------------
#define MAX_POINT_LIGHTS 4

struct SceneUniforms
{
    GLint model, view, projection, normalMatrix, lightSpace, uvScale;
    GLint albedoTex, albedo, alpha, specStrength, shininess, emissive, emissiveTex, doubleSided;
    GLint viewPos, ambientSky, ambientGround, moonDir, moonColor, shadowMap;
    GLint numPointLights, pointPos[MAX_POINT_LIGHTS], pointColor[MAX_POINT_LIGHTS], pointDir[MAX_POINT_LIGHTS], pointAtt[MAX_POINT_LIGHTS];
    GLint resolution, exposure;
};

static SceneUniforms GetSceneUniforms(const Shader& s)
{
    SceneUniforms u{};
    u.model = s.GetUniformLocation("model");
    u.view = s.GetUniformLocation("view");
    u.projection = s.GetUniformLocation("projection");
    u.normalMatrix = s.GetUniformLocation("normalMatrix");
    u.lightSpace = s.GetUniformLocation("lightSpace");
    u.uvScale = s.GetUniformLocation("uvScale");
    u.albedoTex = s.GetUniformLocation("albedoTex");
    u.albedo = s.GetUniformLocation("albedo");
    u.alpha = s.GetUniformLocation("alpha");
    u.specStrength = s.GetUniformLocation("specStrength");
    u.shininess = s.GetUniformLocation("shininess");
    u.emissive = s.GetUniformLocation("emissive");
    u.emissiveTex = s.GetUniformLocation("emissiveTex");
    u.doubleSided = s.GetUniformLocation("doubleSided");
    u.viewPos = s.GetUniformLocation("viewPos");
    u.ambientSky = s.GetUniformLocation("ambientSky");
    u.ambientGround = s.GetUniformLocation("ambientGround");
    u.moonDir = s.GetUniformLocation("moonDir");
    u.moonColor = s.GetUniformLocation("moonColor");
    u.shadowMap = s.GetUniformLocation("shadowMap");
    u.numPointLights = s.GetUniformLocation("numPointLights");
    for (int i = 0; i < MAX_POINT_LIGHTS; ++i)
    {
        std::string idx = "[" + std::to_string(i) + "]";
        u.pointPos[i] = s.GetUniformLocation(("pointPos" + idx).c_str());
        u.pointColor[i] = s.GetUniformLocation(("pointColor" + idx).c_str());
        u.pointDir[i] = s.GetUniformLocation(("pointDir" + idx).c_str());
        u.pointAtt[i] = s.GetUniformLocation(("pointAtt" + idx).c_str());
    }
    u.resolution = s.GetUniformLocation("resolution");
    u.exposure = s.GetUniformLocation("exposure");
    return u;
}

struct PointLight { vec3 pos, colour, dir, att; };

static void DrawScene(const SceneUniforms& u, bool transparentPass, bool cursorOn)
{
    for (const DrawItem& it : gItems)
    {
        if (it.transparent != transparentPass) continue;
        if (it.tag == TAG_CURSOR && !cursorOn) continue;
        const Material& m = it.mat;
        mat3 nm = glm::transpose(glm::inverse(mat3(it.model)));
        glUniformMatrix4fv(u.model, 1, GL_FALSE, glm::value_ptr(it.model));
        glUniformMatrix3fv(u.normalMatrix, 1, GL_FALSE, glm::value_ptr(nm));
        glUniform2f(u.uvScale, m.uvScale.x, m.uvScale.y);
        glUniform3f(u.albedo, m.albedo.x, m.albedo.y, m.albedo.z);
        glUniform1f(u.alpha, m.alpha);
        glUniform1f(u.specStrength, m.spec);
        glUniform1f(u.shininess, m.shininess);
        glUniform3f(u.emissive, m.emissive.x, m.emissive.y, m.emissive.z);
        glUniform1f(u.emissiveTex, m.emissiveTex);
        glUniform1i(u.doubleSided, m.doubleSided ? 1 : 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, m.tex ? m.tex : gTex.white);
        it.mesh->RenderMesh();
    }
}

static void DrawShadowCasters(GLint uModel)
{
    for (const DrawItem& it : gItems)
    {
        if (!it.castShadow || it.transparent) continue;
        glUniformMatrix4fv(uModel, 1, GL_FALSE, glm::value_ptr(it.model));
        it.mesh->RenderMesh();
    }
}

static bool SaveScreenshot(GLuint fbo)
{
    std::vector<unsigned char> pixels(WIDTH * HEIGHT * 3);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    stbi_flip_vertically_on_write(1);
    bool ok = stbi_write_png(SCREENSHOT_NAME, WIDTH, HEIGHT, 3, pixels.data(), WIDTH * 3) != 0;
    if (ok) std::cout << "Saved screenshot: " << std::filesystem::absolute(SCREENSHOT_NAME).u8string() << "\n";
    else    std::cerr << "Failed to save screenshot\n";
    return ok;
}

// ------------------------------------------------------------------------------------
//  Camera
// ------------------------------------------------------------------------------------
const vec3 SHOT_EYE(1.30f, 1.38f, 0.40f);       // มุมกล้องหลักของภาพ
const vec3 SHOT_TARGET(-0.20f, 0.95f, -0.95f);
const float FOV_DEG = 45.0f;

struct Camera
{
    vec3 pos;
    float yaw, pitch;
    void LookAt(const vec3& eye, const vec3& target)
    {
        pos = eye;
        vec3 d = glm::normalize(target - eye);
        yaw = glm::degrees(std::atan2(d.z, d.x));
        pitch = glm::degrees(std::asin(d.y));
    }
    vec3 Front() const
    {
        return glm::normalize(vec3(std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch)),
                                   std::sin(glm::radians(pitch)),
                                   std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch))));
    }
};

// ------------------------------------------------------------------------------------
//  main
// ------------------------------------------------------------------------------------
int main()
{
    Window mainWindow(WIDTH, HEIGHT, 3, 3);
    if (mainWindow.initialise() != 0) return 1;
    GLFWwindow* win = mainWindow.getWindow();
    glfwSetWindowTitle(win, "FUTURE ME  |  P = save image   RMB+mouse = look   WASD/QE = move   R = reset");
    std::cout << "OpenGL " << glGetString(GL_VERSION) << " | " << glGetString(GL_RENDERER) << "\n";

    // ---- shaders ----
    Shader sceneShader, shadowShader;
    sceneShader.CreateFromFiles(shaderDirectory / "futureme_scene.vert", shaderDirectory / "futureme_scene.frag");
    shadowShader.CreateFromFiles(shaderDirectory / "futureme_shadow.vert", shaderDirectory / "futureme_shadow.frag");
    const SceneUniforms su = GetSceneUniforms(sceneShader);
    const GLint shModel = shadowShader.GetUniformLocation("model");
    const GLint shLight = shadowShader.GetUniformLocation("lightSpace");
    if (su.model < 0 || shModel < 0)
    {
        std::cerr << "Shader error: check that the Shaders folder contains the futureme_*.vert/.frag files\n";
        return 1;
    }

    // ---- textures ----
    gTex.white    = MakeWhiteTexture();
    gTex.wood     = MakeWoodTexture();
    gTex.floor    = MakeFloorTexture();
    gTex.wall     = MakeWallTexture();
    gTex.fabric   = MakeFabricTexture();
    gTex.paper    = MakePaperTexture();
    gTex.screen   = MakeScreenTexture();
    gTex.calendar = MakeCalendarTexture();
    gTex.poster   = MakePosterTexture();
    gTex.clock    = MakeClockTexture();
    gTex.can      = MakeCanTexture();
    gTex.sticky1  = MakeStickyTexture(vec3(1.0f, 0.92f, 0.35f), { "DEADLINE", "19 OCT", "2026" }, 3);
    gTex.sticky2  = MakeStickyTexture(vec3(1.0f, 0.55f, 0.70f), { "JUST ONE", "MORE", "RUN..." }, 3);

    // ---- base meshes ----
    { MeshBuilder b; b.Box(mat4(1.0f));    gCube   = Keep(b.Build()); }
    { MeshBuilder b; b.Sphere(mat4(1.0f), 24, 16); gSphere = Keep(b.Build()); }
    { MeshBuilder b; b.Quad(mat4(1.0f));   gQuad   = Keep(b.Build()); }

    // ---- build the scene ----
    BuildRoom();
    BuildDesk();
    BuildComputer();
    BuildChair();
    BuildSkeleton();
    BuildProps();
    BuildCobwebs();
    size_t triCount = 0;
    for (Mesh* m : gMeshes) triCount += m->GetIndexCount() / 3;
    std::cout << "Scene: " << gItems.size() << " draw items, " << triCount << " unique triangles\n";

    // ---- shadow map FBO ----
    GLuint shadowFBO = 0, shadowTex = 0;
    glGenTextures(1, &shadowTex);
    glBindTexture(GL_TEXTURE_2D, shadowTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_SIZE, SHADOW_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    const float border[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
    glGenFramebuffers(1, &shadowFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) std::cerr << "Shadow FBO incomplete\n";

    // ---- MSAA FBO (800x600) + resolve FBO (ใช้ทั้งแสดงผลและบันทึกภาพ) ----
    GLint maxSamples = 1;
    glGetIntegerv(GL_MAX_SAMPLES, &maxSamples);
    const int samples = std::max(1, std::min(4, (int)maxSamples));
    GLuint msFBO = 0, msColor = 0, msDepth = 0, resolveFBO = 0, resolveColor = 0;
    glGenFramebuffers(1, &msFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, msFBO);
    // ระบุ color attachment ให้ชัดเจน เพื่อให้การ render และ blit ทำงานเหมือนกันทุกไดรเวอร์
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glGenRenderbuffers(1, &msColor);
    glBindRenderbuffer(GL_RENDERBUFFER, msColor);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, WIDTH, HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msColor);
    glGenRenderbuffers(1, &msDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, msDepth);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, msDepth);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "MSAA FBO incomplete; cannot render scene.\n";
        return 1;
    }

    glGenFramebuffers(1, &resolveFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, resolveFBO);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glGenRenderbuffers(1, &resolveColor);
    glBindRenderbuffer(GL_RENDERBUFFER, resolveColor);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, WIDTH, HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, resolveColor);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        std::cerr << "Resolve FBO incomplete; cannot display scene.\n";
        return 1;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ---- lights ----
    const vec3 moonDir = glm::normalize(vec3(-1.0f, -0.32f, -0.30f));
    const vec3 moonColour = vec3(0.50f, 0.62f, 1.0f) * 1.45f;
    const vec3 lightCentre(0.2f, 1.2f, 0.6f);
    const mat4 lightView = glm::lookAt(lightCentre - moonDir * 6.0f, lightCentre, Y_AXIS);
    const mat4 lightProj = glm::ortho(-3.6f, 3.6f, -3.2f, 3.2f, 0.5f, 14.0f);
    const mat4 lightSpace = lightProj * lightView;

    const mat4 projection = glm::perspective(glm::radians(FOV_DEG), (float)WIDTH / (float)HEIGHT, 0.05f, 30.0f);

    Camera cam;
    cam.LookAt(SHOT_EYE, SHOT_TARGET);
    double lastX = 0.0, lastY = 0.0;
    bool rmbHeld = false, pWasDown = false;
    float lastTime = (float)glfwGetTime();

    while (!mainWindow.getShouldClose())
    {
        float now = (float)glfwGetTime();
        float dt = now - lastTime;
        lastTime = now;
        glfwPollEvents();

        // ---------------- input ----------------
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(win, GLFW_TRUE);
        if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) cam.LookAt(SHOT_EYE, SHOT_TARGET);
        double mx, my;
        glfwGetCursorPos(win, &mx, &my);
        bool rmb = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if (rmb && rmbHeld)
        {
            cam.yaw += (float)(mx - lastX) * 0.12f;
            cam.pitch = glm::clamp(cam.pitch - (float)(my - lastY) * 0.12f, -89.0f, 89.0f);
        }
        rmbHeld = rmb;
        lastX = mx; lastY = my;
        vec3 front = cam.Front();
        vec3 right = glm::normalize(glm::cross(front, Y_AXIS));
        float speed = 1.5f * dt;
        if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) cam.pos += front * speed;
        if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) cam.pos -= front * speed;
        if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) cam.pos -= right * speed;
        if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) cam.pos += right * speed;
        if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) cam.pos -= Y_AXIS * speed;
        if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) cam.pos += Y_AXIS * speed;
        bool pDown = glfwGetKey(win, GLFW_KEY_P) == GLFW_PRESS;
        bool wantShot = pDown && !pWasDown;
        pWasDown = pDown;

        const mat4 view = glm::lookAt(cam.pos, cam.pos + front, Y_AXIS);

        // ---------------- pass 1 : shadow map ----------------
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
        glViewport(0, 0, SHADOW_SIZE, SHADOW_SIZE);
        glEnable(GL_DEPTH_TEST);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(2.0f, 4.0f);
        shadowShader.UseShader();
        glUniformMatrix4fv(shLight, 1, GL_FALSE, glm::value_ptr(lightSpace));
        DrawShadowCasters(shModel);
        glDisable(GL_POLYGON_OFFSET_FILL);

        // ---------------- pass 2 : scene (MSAA) ----------------
        glBindFramebuffer(GL_FRAMEBUFFER, msFBO);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glViewport(0, 0, WIDTH, HEIGHT);
        // Reset render state in case a previous pass or driver left it changed.
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDepthMask(GL_TRUE);
        glClearColor(0.015f, 0.020f, 0.035f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        sceneShader.UseShader();
        glUniformMatrix4fv(su.view, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(su.projection, 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(su.lightSpace, 1, GL_FALSE, glm::value_ptr(lightSpace));
        glUniform3f(su.viewPos, cam.pos.x, cam.pos.y, cam.pos.z);
        glUniform3f(su.ambientSky, 0.14f, 0.15f, 0.20f);
        glUniform3f(su.ambientGround, 0.07f, 0.06f, 0.06f);
        glUniform3f(su.moonDir, moonDir.x, moonDir.y, moonDir.z);
        glUniform3f(su.moonColor, moonColour.x, moonColour.y, moonColour.z);
        glUniform2f(su.resolution, (float)WIDTH, (float)HEIGHT);
        glUniform1f(su.exposure, 1.35f);
        glUniform1i(su.albedoTex, 0);
        glUniform1i(su.shadowMap, 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, shadowTex);

        // แสงจากจอ (กะพริบเบาๆ) + ไฟ LED เคส
        float flicker = 1.0f + 0.05f * std::sin(now * 23.0f) * std::sin(now * 3.1f);
        const PointLight lights[2] = {
            { gScene.screenCentre + gScene.screenNormal * 0.18f, vec3(1.0f, 0.22f, 0.18f) * 2.6f * flicker, gScene.screenNormal, vec3(1.0f, 0.6f, 1.6f) },
            { gScene.pcLed, vec3(0.25f, 0.45f, 1.0f) * 0.6f, vec3(0.0f), vec3(1.0f, 3.0f, 10.0f) } };
        glUniform1i(su.numPointLights, 2);
        for (int i = 0; i < 2; ++i)
        {
            glUniform3fv(su.pointPos[i], 1, glm::value_ptr(lights[i].pos));
            glUniform3fv(su.pointColor[i], 1, glm::value_ptr(lights[i].colour));
            glUniform3fv(su.pointDir[i], 1, glm::value_ptr(lights[i].dir));
            glUniform3fv(su.pointAtt[i], 1, glm::value_ptr(lights[i].att));
        }

        bool cursorOn = std::fmod(now, 1.0f) < 0.55f;
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        DrawScene(su, false, cursorOn);                 // วัตถุทึบ
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        DrawScene(su, true, cursorOn);                  // ใยแมงมุม (โปร่งแสง)
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);

        // ---------------- resolve MSAA -> แสดงบนหน้าต่าง ----------------
        glBindFramebuffer(GL_READ_FRAMEBUFFER, msFBO);
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolveFBO);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT, GL_COLOR_BUFFER_BIT, GL_NEAREST);

        if (wantShot) SaveScreenshot(resolveFBO);

        int fbw = 0, fbh = 0;
        glfwGetFramebufferSize(win, &fbw, &fbh);
        if (fbw > 0 && fbh > 0)
        {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, resolveFBO);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glDrawBuffer(GL_BACK);
            glBlitFramebuffer(0, 0, WIDTH, HEIGHT, 0, 0, fbw, fbh, GL_COLOR_BUFFER_BIT, GL_LINEAR);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(0);
        mainWindow.swapBuffers();
    }

    // ---- cleanup ----
    for (Mesh* m : gMeshes) delete m;
    gMeshes.clear();
    gItems.clear();
    GLuint texList[] = { gTex.white, gTex.wood, gTex.floor, gTex.wall, gTex.fabric, gTex.screen, gTex.calendar,
                         gTex.poster, gTex.sticky1, gTex.sticky2, gTex.clock, gTex.can, gTex.paper, shadowTex };
    glDeleteTextures((GLsizei)(sizeof(texList) / sizeof(GLuint)), texList);
    glDeleteRenderbuffers(1, &msColor);
    glDeleteRenderbuffers(1, &msDepth);
    glDeleteRenderbuffers(1, &resolveColor);
    glDeleteFramebuffers(1, &msFBO);
    glDeleteFramebuffers(1, &resolveFBO);
    glDeleteFramebuffers(1, &shadowFBO);
    return 0;
}
