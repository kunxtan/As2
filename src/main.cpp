#undef GLFW_DLL

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <filesystem>
#include <vector>
#include <iostream>

#include "Libs/Mesh.h"
#include "Libs/Shader.h"
#include "Libs/Window.h"
#include "ProjectPaths.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// แก้ตัวสะกด IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include <stb_image.h>

const GLint WIDTH = 800;
const GLint HEIGHT = 600;

std::vector<Mesh*> meshList;
std::vector<Shader*> shaderList;
Mesh* lightMesh = nullptr; // cube that shows where the light is

const std::filesystem::path shaderDirectory =
    std::filesystem::u8path(OPENGL_STARTER_SHADER_DIR);
const std::filesystem::path textureDirectory =
    std::filesystem::u8path(OPENGL_STARTER_TEXTURE_DIR);
const std::filesystem::path modelDirectory =
    std::filesystem::u8path(OPENGL_STARTER_MODEL_DIR);

void CreateTriangle()
{
    GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         0.0f, -1.0f, 1.0f, 0.5f, 0.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
         0.0f,  1.0f, 0.0f, 0.5f, 1.0f
    };

    unsigned int indices[] = {
        0, 3, 1,
        1, 3, 2,
        2, 3, 0,
        0, 1, 2
    };

    Mesh* obj1 = new Mesh();
    // indices มีทั้งหมด 12 ตัว
    obj1->CreateMesh(vertices, indices, 20, 12);
    
    for (int i = 0; i < 10; i++)
    {
        meshList.push_back(obj1);
    }
}

void CreateOBJ(const std::filesystem::path& modelPath)
{
    Mesh* obj = new Mesh();
    if (!obj->CreateMeshFromOBJ(modelPath.u8string()))
    {
        std::cerr << "Failed to load OBJ: " << modelPath << "\n";
        delete obj;
        return;
    }
    
    // ใส่ pointer ตัวเดิมเข้าไป 10 ครั้ง เพื่อให้สอดคล้องกับลูป render 10 ชิ้น
    for (int i = 0; i < 10; i++)
    {
        meshList.push_back(obj);
    }
}

// ปรับชื่อเป็น LoadTexture (T ตัวใหญ่)
GLuint LoadTexture(const std::filesystem::path& texturePath)
{
    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char* data = stbi_load(texturePath.u8string().c_str(), &width, &height, &channels, 0);

    if (data == nullptr)
    {
        std::cerr << "Failed to load texture: " << stbi_failure_reason() << "\n";
        return 0;
    }

    GLenum format = GL_RGB;
    switch (channels)
    {
    case 4: format = GL_RGBA; break;
    case 3: format = GL_RGB; break;
    case 2: format = GL_RG; break;
    case 1: format = GL_RED; break;
    default:
        std::cerr << "Unsupported texture format\n";
        stbi_image_free(data);
        return 0;
    }

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);
    glBindTexture(GL_TEXTURE_2D, 0);

    return texture;
}

void CreateShaders()
{
    Shader* shader1 = new Shader();
    shader1->CreateFromFiles(
        shaderDirectory / "shader.vert",
        shaderDirectory / "shader.frag"
    );
    shaderList.push_back(shader1);

    // shaderList[1]: simple shader for drawing the light cube
    Shader* shader2 = new Shader();
    shader2->CreateFromFiles(
        shaderDirectory / "light_shader.vert",
        shaderDirectory / "light_shader.frag"
    );
    shaderList.push_back(shader2);
}

void CreateLightCube(const std::filesystem::path& modelPath)
{
    lightMesh = new Mesh();
    if (!lightMesh->CreateMeshFromOBJ(modelPath.u8string()))
    {
        std::cerr << "Failed to load OBJ: " << modelPath << "\n";
        delete lightMesh;
        lightMesh = nullptr;
    }
}

void Cleanup()
{
    if (!meshList.empty()) {
        delete meshList[0];
    }
    meshList.clear();

    delete lightMesh;
    lightMesh = nullptr;

    for (Shader* shader : shaderList)
        delete shader;
    shaderList.clear();
}

int main()
{
    Window mainWindow(WIDTH, HEIGHT, 3, 3);
    if (mainWindow.initialise() != 0) return 1;

    // CreateTriangle();
    CreateOBJ(modelDirectory / "suzanne.obj");
    CreateShaders();
    CreateLightCube(modelDirectory / "cube.obj");

    // หมายเหตุ: สำหรับ Task ให้เปลี่ยนเป็น "cloth.jpg"
    GLuint texture_data = LoadTexture(textureDirectory / "uvmap.png");

    if (texture_data == 0) {
        return 1;
    }

    glm::mat4 view(1.0f);
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
    glm::vec3 cameraTarget = glm::vec3(-1.0f, 1.0f, -1.0f);
    glm::vec3 cameraDirection = glm::normalize(cameraTarget - cameraPos);

    float yaw = -90.0f;
    float pitch = 0.0f;
    cameraDirection.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraDirection.y = sin(glm::radians(pitch));
    cameraDirection.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraDirection = glm::normalize(cameraDirection);

    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    double lastX = 400.0;
    double lastY = 300.0;
    bool firstMouse = true;
    float mouseSensitivity = 0.05f;

    glm::vec3 pyramidPositions[] =
    {
        glm::vec3(0.0f,  0.0f, -2.5f),
        glm::vec3(2.0f,  5.0f, -15.0f),
        glm::vec3(-1.5f, -2.2f, -2.5f),
        glm::vec3(-3.8f, -2.0f, -12.3f),
        glm::vec3(2.4f, -0.4f, -3.5f),
        glm::vec3(-1.7f,  3.0f, -7.5f),
        glm::vec3(1.3f, -2.0f, -2.5f),
        glm::vec3(1.5f,  2.0f, -2.5f),
        glm::vec3(1.5f,  0.2f, -1.5f),
        glm::vec3(-1.3f,  1.0f, -1.5f)
    };

    glm::mat4 projection = glm::perspective(
        glm::radians(45.0f), 
        (GLfloat)mainWindow.getBufferWidth() / (GLfloat)mainWindow.getBufferHeight(), 
        0.1f, 100.0f
    );

    float lastTime = static_cast<float>(glfwGetTime());
    glm::vec3 lightColour = glm::vec3(1.0f, 1.0f, 1.0f); // white light
    glm::vec3 lightPos = glm::vec3(5.0f, 5.0f, 0.0f);

    while (!mainWindow.getShouldClose())
    {
        float currentTime = static_cast<float>(glfwGetTime());
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        glfwPollEvents();

        double mouseX, mouseY;
        glfwGetCursorPos(mainWindow.getWindow(), &mouseX, &mouseY);

        if (firstMouse) {
            lastX = mouseX;
            lastY = mouseY;
            firstMouse = false;
        }

        double xoffset = (mouseX - lastX) * mouseSensitivity;
        double yoffset = (lastY - mouseY) * mouseSensitivity;
        lastX = mouseX;
        lastY = mouseY;

        yaw += static_cast<float>(xoffset);
        pitch += static_cast<float>(yoffset);

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        cameraDirection.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraDirection.y = sin(glm::radians(pitch));
        cameraDirection.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraDirection = glm::normalize(cameraDirection);

        glm::vec3 cameraRight = glm::normalize(glm::cross(cameraDirection, up));
        glm::vec3 cameraUp = glm::normalize(glm::cross(cameraRight, cameraDirection));

        float moveSpeed = 5.0f;
        if (glfwGetKey(mainWindow.getWindow(), GLFW_KEY_W) == GLFW_PRESS)
            cameraPos += cameraDirection * moveSpeed * deltaTime;
        if (glfwGetKey(mainWindow.getWindow(), GLFW_KEY_S) == GLFW_PRESS)
            cameraPos -= cameraDirection * moveSpeed * deltaTime;
        if (glfwGetKey(mainWindow.getWindow(), GLFW_KEY_A) == GLFW_PRESS)
            cameraPos -= cameraRight * moveSpeed * deltaTime;
        if (glfwGetKey(mainWindow.getWindow(), GLFW_KEY_D) == GLFW_PRESS)
            cameraPos += cameraRight * moveSpeed * deltaTime;

        // Space bar: reset mouse look and camera position
        if (glfwGetKey(mainWindow.getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS)
        {
            yaw = -90.0f;
            pitch = 0.0f;
            cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
        }

        // Moving light: x swings left-right, y moves up-down, z stays fixed
        lightPos.x = 1.0f + sin(static_cast<float>(glfwGetTime())) * 2.0f;
        lightPos.y = sin(static_cast<float>(glfwGetTime()) / 2.0f) * 1.0f;

        view = glm::lookAt(cameraPos, cameraPos + cameraDirection, cameraUp);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shaderList[0]->UseShader();
        
        GLuint uniformModel = shaderList[0]->GetUniformLocation("model");
        GLuint uniformView = shaderList[0]->GetUniformLocation("view");
        GLuint uniformProjection = shaderList[0]->GetUniformLocation("projection");
        GLuint uniformTexture1 = shaderList[0]->GetUniformLocation("texture1");

        // กำหนด sampler ให้ชี้ไปที่ Texture Unit 0
        glUniform1i(uniformTexture1, 0);

        // Bind Texture เข้า Unit 0 และปล่อยให้คงอยู่ขณะ Render
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_data);
        
        glUniform3fv(shaderList[0]->GetUniformLocation("lightColour"), 1, (GLfloat*)&lightColour);
        glUniform3fv(shaderList[0]->GetUniformLocation("lightPos"), 1, (GLfloat*)&lightPos);
        glUniform3fv(shaderList[0]->GetUniformLocation("viewPos"), 1, (GLfloat*)&cameraPos);
        glm::mat4 model (1.0f);

        glUniformMatrix4fv(uniformView, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(uniformProjection, 1, GL_FALSE, glm::value_ptr(projection));

        for (int i = 0; i < 10; i++)
        {
            glm::mat4 model(1.0f);
            model = glm::translate(model, pyramidPositions[i]);
            model = glm::rotate(model, glm::radians(2.0f * i), glm::vec3(1.0f, 0.3f, 0.5f));
            model = glm::scale(model, glm::vec3(0.8f, 0.8f, 1.0f));
            
            glUniformMatrix4fv(uniformModel, 1, GL_FALSE, glm::value_ptr(model));
            meshList[i]->RenderMesh();
        }

        // Draw the light cube with the light shader
        if (lightMesh != nullptr)
        {
            shaderList[1]->UseShader();
            glm::mat4 lightModel(1.0f);
            lightModel = glm::translate(lightModel, lightPos);
            lightModel = glm::scale(lightModel, glm::vec3(0.2f));

            glUniformMatrix4fv(shaderList[1]->GetUniformLocation("model"), 1, GL_FALSE, glm::value_ptr(lightModel));
            glUniformMatrix4fv(shaderList[1]->GetUniformLocation("view"), 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(shaderList[1]->GetUniformLocation("projection"), 1, GL_FALSE, glm::value_ptr(projection));
            glUniform3fv(shaderList[1]->GetUniformLocation("lightColour"), 1, (GLfloat*)&lightColour);
            lightMesh->RenderMesh();
        }

        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
        mainWindow.swapBuffers();
    }

    glDeleteTextures(1, &texture_data);
    Cleanup();
    return 0;
}