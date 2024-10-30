
////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////
#include <cstddef>
#include <cstring>
#include <stdlib.h>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#define GLEW_STATIC
#include "GL/glew.h"
#include "GL/glfw3.h"
#include "arcball.h"
#include "cvec.h"
#include "geometry.h"
#include "geometrymaker.h"
#include "glsupport.h"
#include "matrix4.h"
#include "ppm.h"
#include "rigtform.h"
#include "scenegraph.h"
#include "sgutils.h"
#include "asstcommon.h"
#include "drawer.h"
#include "picker.h"
using namespace std;
// G L O B A L S ///////////////////////////////////////////////////
static const float g_frustMinFov = 60.0; // A minimal of 60 degree field of view
static float g_frustFovY =
    g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)
static const float g_frustNear = -0.1;  // near plane
static const float g_frustFar = -50.0;  // far plane
static const float g_groundY = -2.0;    // y coordinate of the ground
static const float g_groundSize = 10.0; // half the ground length
enum SkyMode { WORLD_SKY = 0, SKY_SKY = 1 };
static GLFWwindow *g_window;
static int g_windowWidth = 512;
static int g_windowHeight = 512;
static double g_wScale = 1;
static double g_hScale = 1;
static bool g_mouseClickDown = false; // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static bool g_spaceDown = false; // space state, for middle mouse emulation
static double g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;
static SkyMode g_activeCameraFrame = WORLD_SKY;
static bool g_displayArcball = true;
static double g_arcballScreenRadius = 100; // number of pixels
static double g_arcballScale = 1;
static bool g_pickingMode = false;
static bool g_playingAnimation = true;
// --------- Materials
static shared_ptr<Material> g_planetMat, g_astMat, g_sunMat, g_bumpFloorMat,
    g_arcballMat, g_pickingMat, g_lightMat;
shared_ptr<Material> g_overridingMaterial;
// --------- Geometry
typedef SgGeometryShapeNode MyShapeNode;
// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;
// --------- Scene
static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_stars, g_robot1Node, g_solarSystem, g_asteroidBelt,
    g_kuiperBelt, g_light1
//    ,g_light2
;
static shared_ptr<SgRbtNode> g_currentCameraNode;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode;
static double g_lastFrameClock;
static int g_framesPerSecond = 60; // frames to render per second during animation playback
/// PLANET GLOBALS
///
static RigTForm initSkyRbt = RigTForm(Cvec3(-1.12, 4.5, -5.91), Quat(.0419, -0.0166, -0.65599, -0.26));
struct PlanetInfo {
    // FROM https://nssdc.gsfc.nasa.gov/planetary/factsheet/
    float diameter;
    float distance;
    float inclination;
    float period;
    float theta_at_peak;
};
static PlanetInfo mercury = {3879, 57.9, 7, 88, 1};
static PlanetInfo venus = {12104, 108.2, 3.4, 224.7, 2};
static PlanetInfo earth = {12756, 149.6, 0, 365.2, 3};
static PlanetInfo mars = {6792, 228, 1.8, 687, 4};
static PlanetInfo jupiter = {142984, 778.5, 1.3, 4331, 5};
static PlanetInfo saturn = {120536, 1432, 2.5, 10747, 6};
static PlanetInfo uranus = {51118, 2867, .8, 30589, 7};
static PlanetInfo neptune = {49528, 4515, 2.8, 59800, 8};
static PlanetInfo pluto = {2376, 5906, 17.2, 90560, 9};

const static int NUM_PLANETS = 9;
static PlanetInfo planetData[NUM_PLANETS] = {mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, pluto};
static shared_ptr<SgTransformNode> jointNodes[NUM_PLANETS + 1];
static float timeRatio = .0002765 * earth.period; // 1 minute per earthyear
static int currentPlanetYear = 2;
static bool toScale = false;
static bool inLine = false;
static string planetNames[NUM_PLANETS] = {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"
};
static shared_ptr<Material> g_mercMat, g_venusMat, g_earthMat, g_marsMat, g_jupiterMat, g_saturnMat, g_neptuneMat, g_uranusMat, g_plutoMat, g_asteroidMat;

default_random_engine generator;
normal_distribution<double> distribution(0.0,1.0);
///////////////// END OF G L O B A L S /////////////////////////////////////////
///


static void initGround() {
    int ibLen, vbLen;
    getPlaneVbIbLen(vbLen, ibLen);
    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);
    makePlane(g_groundSize * 2, vtx.begin(), idx.begin());
    g_ground.reset(
        new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}
static void initCubes() {
    int ibLen, vbLen;
    getCubeVbIbLen(vbLen, ibLen);
    // Temporary storage for cube Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);
    makeCube(1, vtx.begin(), idx.begin());
    g_cube.reset(
        new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vbLen, ibLen));
}
static void initSphere() {
    int ibLen, vbLen;
    getSphereVbIbLen(20, 10, vbLen, ibLen);
    // Temporary storage for sphere Geometry
    vector<VertexPNTBX> vtx(vbLen);
    vector<unsigned short> idx(ibLen);
    makeSphere(1, 20, 10, vtx.begin(), idx.begin());
    g_sphere.reset(new SimpleIndexedGeometryPNTBX(&vtx[0], &idx[0], vtx.size(),
                                                  idx.size()));
}
static void sendProjectionMatrix(Uniforms &uniforms,
                                 const Matrix4 &projMatrix) {
    uniforms.put("uProjMatrix", projMatrix);
}
// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
    if (g_windowWidth >= g_windowHeight)
        g_frustFovY = g_frustMinFov;
    else {
        const double RAD_PER_DEG = 0.5 * CS175_PI / 180;
        g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight /
                                g_windowWidth,
                            cos(g_frustMinFov * RAD_PER_DEG)) /
                      RAD_PER_DEG;
    }
}
static Matrix4 makeProjectionMatrix() {
    return Matrix4::makeProjection(
        g_frustFovY, g_windowWidth / static_cast<double>(g_windowHeight),
        g_frustNear, g_frustFar);
}
enum ManipMode { ARCBALL_ON_PICKED, ARCBALL_ON_SKY, EGO_MOTION };
static ManipMode getManipMode() {
    // if nothing is picked or the picked transform is the transfrom we are
    // viewing from
    if (g_currentPickedRbtNode == NULL ||
        g_currentPickedRbtNode == g_currentCameraNode) {
        if (g_currentCameraNode == g_skyNode &&
            g_activeCameraFrame == WORLD_SKY)
            return ARCBALL_ON_SKY;
        else
            return EGO_MOTION;
    } else
        return ARCBALL_ON_PICKED;
}
static bool shouldUseArcball() {
    return (g_currentPickedRbtNode != 0);
    //  return getManipMode() != EGO_MOTION;
}
// The translation part of the aux frame either comes from the current
// active object, or is the identity matrix when
static RigTForm getArcballRbt() {
    switch (getManipMode()) {
    case ARCBALL_ON_PICKED:
        return getPathAccumRbt(g_world, g_currentPickedRbtNode);
    case ARCBALL_ON_SKY:
        return RigTForm();
    case EGO_MOTION:
        return getPathAccumRbt(g_world, g_currentCameraNode);
    default:
        throw runtime_error("Invalid ManipMode");
    }
}
static void updateArcballScale() {
    RigTForm arcballEye =
        inv(getPathAccumRbt(g_world, g_currentCameraNode)) * getArcballRbt();
    double depth = arcballEye.getTranslation()[2];
    if (depth > -CS175_EPS)
        g_arcballScale = 0.02;
    else
        g_arcballScale =
            getScreenToEyeScale(depth, g_frustFovY, g_windowHeight);
}
static void drawArcBall(Uniforms &uniforms) {
    RigTForm arcballEye =
        inv(getPathAccumRbt(g_world, g_currentCameraNode)) * getArcballRbt();
    Matrix4 MVM = rigTFormToMatrix(arcballEye) *
                  Matrix4::makeScale(Cvec3(1, 1, 1) * g_arcballScale *
                                     g_arcballScreenRadius);
    sendModelViewNormalMatrix(uniforms, MVM, normalMatrix(MVM));
    g_arcballMat->draw(*g_sphere, uniforms);
}
static void drawStuff(bool picking) {
    // if we are not translating, update arcball scale
    if (!(g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton) ||
          (g_mouseLClickButton && !g_mouseRClickButton && g_spaceDown)))
        updateArcballScale();
    Uniforms uniforms;
    // build & send proj. matrix to vshader
    const Matrix4 projmat = makeProjectionMatrix();
    sendProjectionMatrix(uniforms, projmat);
    const RigTForm eyeRbt = getPathAccumRbt(g_world, g_currentCameraNode);
    const RigTForm invEyeRbt = inv(eyeRbt);
    Cvec3 l1 = getPathAccumRbt(g_world, g_light1).getTranslation();
//    Cvec3 l2 = getPathAccumRbt(g_world, g_light2).getTranslation();
    uniforms.put("uLight", Cvec3(invEyeRbt * Cvec4(l1, 1)));
//    uniforms.put("uLight2", Cvec3(invEyeRbt * Cvec4(l2, 1)));
    if (!picking) {
        Drawer drawer(invEyeRbt, uniforms);
        g_world->accept(drawer);
        if (g_displayArcball && shouldUseArcball())
            drawArcBall(uniforms);
    } else {
        Picker picker(invEyeRbt, uniforms);
        g_overridingMaterial = g_pickingMat;
        g_world->accept(picker);
        g_overridingMaterial.reset();
        glFlush();
        g_currentPickedRbtNode =
            picker.getRbtNodeAtXY(g_mouseClickX * g_wScale,
                                  g_mouseClickY * g_hScale);
        if (g_currentPickedRbtNode == g_groundNode)
            g_currentPickedRbtNode.reset(); // set to NULL
        cout << (g_currentPickedRbtNode ? "Part picked" : "No part picked")
             << endl;
    }
}
static void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    drawStuff(false);
    glfwSwapBuffers(g_window);
    checkGlErrors();
}
static void pick() {
    // We need to set the clear color to black, for pick rendering.
    // so let's save the clear color
    GLdouble clearColor[4];
    glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    drawStuff(true);
    // Uncomment below to see result of the pick rendering pass
    //glfwSwapBuffers(g_window);
    // Now set back the clear color
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
    checkGlErrors();
}
static void updatePlanets();
static void animationUpdate() {
    if (g_playingAnimation) {
        updatePlanets();
    }
}
static void reshape(GLFWwindow * window, const int w, const int h) {
    int width, height;
    glfwGetFramebufferSize(g_window, &width, &height);
    glViewport(0, 0, width, height);
    
    g_windowWidth = w;
    g_windowHeight = h;
    cerr << "Size of window is now " << g_windowWidth << "x" << g_windowHeight << endl;
    g_arcballScreenRadius = max(1.0, min(h, w) * 0.25);
    updateFrustFovY();
}
static Cvec3 getArcballDirection(const Cvec2 &p, const double r) {
    double n2 = norm2(p);
    if (n2 >= r * r)
        return normalize(Cvec3(p, 0));
    else
        return normalize(Cvec3(p, sqrt(r * r - n2)));
}
static RigTForm moveArcball(const Cvec2 &p0, const Cvec2 &p1) {
    const Matrix4 projMatrix = makeProjectionMatrix();
    const RigTForm eyeInverse =
        inv(getPathAccumRbt(g_world, g_currentCameraNode));
    const Cvec3 arcballCenter = getArcballRbt().getTranslation();
    const Cvec3 arcballCenter_ec = Cvec3(eyeInverse * Cvec4(arcballCenter, 1));
    if (arcballCenter_ec[2] > -CS175_EPS)
        return RigTForm();
    Cvec2 ballScreenCenter =
        getScreenSpaceCoord(arcballCenter_ec, projMatrix, g_frustNear,
                            g_frustFovY, g_windowWidth, g_windowHeight);
    const Cvec3 v0 =
        getArcballDirection(p0 - ballScreenCenter, g_arcballScreenRadius);
    const Cvec3 v1 =
        getArcballDirection(p1 - ballScreenCenter, g_arcballScreenRadius);
    return RigTForm(Quat(0.0, v1[0], v1[1], v1[2]) *
                    Quat(0.0, -v0[0], -v0[1], -v0[2]));
}
static RigTForm doMtoOwrtA(const RigTForm &M, const RigTForm &O,
                           const RigTForm &A) {
    return A * M * inv(A) * O;
}
static RigTForm getMRbt(const double dx, const double dy) {
    RigTForm M;
    if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) {
        if (shouldUseArcball())
            M = moveArcball(Cvec2(g_mouseClickX, g_mouseClickY),
                            Cvec2(g_mouseClickX + dx, g_mouseClickY + dy));
        else
            M = RigTForm(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
    } else {
        double movementScale =
            getManipMode() == EGO_MOTION ? 0.02 : g_arcballScale;
        if (g_mouseRClickButton && !g_mouseLClickButton) {
            M = RigTForm(Cvec3(dx, dy, 0) * movementScale);
        } else if (g_mouseMClickButton ||
                   (g_mouseLClickButton && g_mouseRClickButton) ||
                   (g_mouseLClickButton && g_spaceDown)) {
            M = RigTForm(Cvec3(0, 0, -dy) * movementScale);
        }
    }
    switch (getManipMode()) {
    case ARCBALL_ON_PICKED:
        break;
    case ARCBALL_ON_SKY:
        M = inv(M);
        break;
    case EGO_MOTION:
        if (g_mouseLClickButton && !g_mouseRClickButton &&
            !g_spaceDown) // only invert rotation
            M = inv(M);
        break;
    }
    return M;
}
static RigTForm makeMixedFrame(const RigTForm &objRbt, const RigTForm &eyeRbt) {
    return transFact(objRbt) * linFact(eyeRbt);
}
// l = w X Y Z
// o = l O
// a = w A = l (Z Y X)^1 A = l A'
// o = a (A')^-1 O
//   => a M (A')^-1 O = l A' M (A')^-1 O
static void motion(GLFWwindow *window, double x, double y) {
    if (!g_mouseClickDown)
        return;
    const double dx = x - g_mouseClickX;
    const double dy = g_windowHeight - y - 1 - g_mouseClickY;
    const RigTForm M = getMRbt(dx, dy); // the "action" matrix
    // the matrix for the auxiliary frame (the w.r.t.)
    RigTForm A = makeMixedFrame(getArcballRbt(),
                                getPathAccumRbt(g_world, g_currentCameraNode));
    shared_ptr<SgRbtNode> target;
    switch (getManipMode()) {
    case ARCBALL_ON_PICKED:
        target = g_currentPickedRbtNode;
        break;
    case ARCBALL_ON_SKY:
        target = g_skyNode;
        break;
    case EGO_MOTION:
        target = g_currentCameraNode;
        break;
    }
    A = inv(getPathAccumRbt(g_world, target, 1)) * A;
    if ((g_mouseLClickButton && !g_mouseRClickButton &&
         !g_spaceDown) // rotating
        && target == g_skyNode) {
        RigTForm My = getMRbt(dx, 0);
        RigTForm Mx = getMRbt(0, dy);
        RigTForm B = makeMixedFrame(getArcballRbt(), RigTForm());
        RigTForm O = doMtoOwrtA(Mx, target->getRbt(), A);
        O = doMtoOwrtA(My, O, B);
        target->setRbt(O);
    } else {
        target->setRbt(doMtoOwrtA(M, target->getRbt(), A));
    }
    g_mouseClickX += dx;
    g_mouseClickY += dy;
}
static void mouse(GLFWwindow *window, int button, int state, int mods) {
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    g_mouseClickX = x;
    g_mouseClickY =
        g_windowHeight - y - 1; // conversion from GLFW window-coordinate-system
                                // to OpenGL window-coordinate-system
    g_mouseLClickButton |= (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS);
    g_mouseRClickButton |= (button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_PRESS);
    g_mouseMClickButton |= (button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_PRESS);
    g_mouseLClickButton &= !(button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_RELEASE);
    g_mouseRClickButton &= !(button == GLFW_MOUSE_BUTTON_RIGHT && state == GLFW_RELEASE);
    g_mouseMClickButton &= !(button == GLFW_MOUSE_BUTTON_MIDDLE && state == GLFW_RELEASE);
    g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;
    if (g_pickingMode && button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS) {
        pick();
        g_pickingMode = false;
        cerr << "Picking mode is off" << endl;
    }
}
static void initScene();
static void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            case GLFW_KEY_SPACE:
                g_spaceDown = true;
                break;
            case GLFW_KEY_ESCAPE:
                exit(0);
            case GLFW_KEY_H:
                cout << " ============== H E L P ==============\n\n"
                << "h\t\thelp menu\n"
                << "\tdrag left mouse to rotate\n"
                << "\tdrag right mouse button to move\n"
                << "s\t\tsave screenshot\n"
                << "f\t\tToggle all motion\n"
                << "v\t\tChange view scale\n"
                << "m\t\tChange view MODE (sky-sky or sky-world)\n"
                << "l\t\tToggle starting planets inline\n"
                << "a\t\tToggle to presets speeds\n"
                << ">\t\tSpeed up time\n"
                << "<\t\tSlow down time\n"
                << "p\t\tPrint info for view (DEBUG)\n"
                << endl;
                break;
            case GLFW_KEY_S:
                glFlush();
                writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
                break;
            case GLFW_KEY_V: {
                toScale = !toScale;
                initScene();
//                initCelestials();
            } break;
            case GLFW_KEY_P: {
                Cvec3 pos = g_skyNode->getRbt().getTranslation();
                Quat q = g_skyNode->getRbt().getRotation();
                cerr << "RBT: \n";
                for(int i = 0; i < 3; i++){
                    cerr<<pos[i] << "  ";
                }
                cerr<<"\n";
                for (int i=0; i < 4; i++){
                    cerr<<q[i]<<"  ";
                } cerr << "\n";
                break;}
                //            g_pickingMode = !g_pickingMode;
                //            cerr << "Picking mode is " << (g_pickingMode ? "on" : "off") << endl;
                //            break;
            case GLFW_KEY_M:
                g_activeCameraFrame = SkyMode((g_activeCameraFrame + 1) % 2);
                cerr << "Editing sky eye w.r.t. "
                << (g_activeCameraFrame == WORLD_SKY ? "world-sky frame\n"
                    : "sky-sky frame\n")
                << endl;
                break;
            case GLFW_KEY_A:
                currentPlanetYear += 1;
                currentPlanetYear %= NUM_PLANETS;
                timeRatio = .0002765 *  planetData[currentPlanetYear].period;
                cerr << "1 minute is now one year on " << planetNames[currentPlanetYear] << "\n";
                break;
            case GLFW_KEY_F:
                g_playingAnimation = !g_playingAnimation;
                if (g_playingAnimation){
                    cerr << "Start\n";
                } else {cerr << "Stop\n";}
                break;
            case GLFW_KEY_R:
                cerr << "Rerandomizing planets\n";
                initScene();
                break;
            case GLFW_KEY_L:
                inLine = !inLine;
                cerr << "Inline is now ";
                if (inLine){ cerr<<"TRUE\n";}
                else {cerr << "FALSE\n";}
                initScene();
                break;
            case GLFW_KEY_D:
                break;
            case GLFW_KEY_PERIOD: // >
                if (!(mods & GLFW_MOD_SHIFT)) break;
                timeRatio *= 1.05;
                break;
            case GLFW_KEY_COMMA: // <
                if (!(mods & GLFW_MOD_SHIFT)) break;
                timeRatio /= 1.05;
                break;
            case GLFW_KEY_W:
                break;
            case GLFW_KEY_I:
                break;
            case GLFW_KEY_MINUS:
                break;
            case GLFW_KEY_EQUAL: // +
                if (!(mods & GLFW_MOD_SHIFT)) break;
                break;
        }
    }
                
        else {
        switch(key) {
        case GLFW_KEY_SPACE:
            g_spaceDown = false;
            break;
        }
    }
}
void error_callback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}
static void initGlfwState() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);
    g_window = glfwCreateWindow(g_windowWidth, g_windowHeight,
                                "Assignment 8", NULL, NULL);
    if (!g_window) {
        fprintf(stderr, "Failed to create GLFW window or OpenGL context\n");
        exit(1);
    }
    glfwMakeContextCurrent(g_window);
    glewInit();
    glfwSwapInterval(1);
    glfwSetErrorCallback(error_callback);
    glfwSetMouseButtonCallback(g_window, mouse);
    glfwSetCursorPosCallback(g_window, motion);
    glfwSetWindowSizeCallback(g_window, reshape);
    glfwSetKeyCallback(g_window, keyboard);
    int screen_width, screen_height;
    glfwGetWindowSize(g_window, &screen_width, &screen_height);
    int pixel_width, pixel_height;
    glfwGetFramebufferSize(g_window, &pixel_width, &pixel_height);
    cout << screen_width << " " << screen_height << endl;
    cout << pixel_width << " " << pixel_width << endl;
    g_wScale = pixel_width / screen_width;
    g_hScale = pixel_height / screen_height;
}
static void initGLState() {
    glClearColor(0.005, 0.005, 0.005, 0.);
    glClearDepth(0.);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glReadBuffer(GL_BACK);
        glEnable(GL_FRAMEBUFFER_SRGB);
}
static void initMaterials() {
    // Create some prototype materials
    Material diffuse("./shaders/basic-gl3.vshader",
                     "./shaders/diffuse-gl3.fshader");
    Material solid("./shaders/basic-gl3.vshader",
                   "./shaders/solid-gl3.fshader");
    
    // copy diffuse prototype and set red color
    g_planetMat.reset(new Material(solid));
    g_planetMat->getUniforms().put("uColor", Cvec3f(1, 0, 0));
    
    // copy diffuse prototype and set blue color
    g_astMat.reset(new Material(solid));
    g_astMat->getUniforms().put("uColor", Cvec3f(0, 0, 1));
    
    // sun material
    g_sunMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_sunMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("sun.ppm", true)));
    g_sunMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("sun.ppm", false)));
    
    // mercury material
    g_mercMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_mercMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("mercury.ppm", true)));
    g_mercMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("mercury.ppm", false)));
    
    // venus material
    g_venusMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_venusMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("venus.ppm", true)));
    g_venusMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("venus.ppm", false)));
    
    // earth material
    g_earthMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_earthMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("earth.ppm", true)));
    g_earthMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("earth.ppm", false)));
    
    // mars material
    g_marsMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_marsMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("mars.ppm", true)));
    g_marsMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("mars.ppm", false)));
    
    // jupiter material
    g_jupiterMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_jupiterMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("jupiter.ppm", true)));
    g_jupiterMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("jupiter.ppm", false)));
    
    // saturn material
    g_saturnMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_saturnMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("saturn.ppm", true)));
    g_saturnMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("saturn.ppm", false)));
    
    // neptune material
    g_neptuneMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_neptuneMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("neptune.ppm", true)));
    g_neptuneMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("neptune.ppm", false)));
    
    // uranus material
    g_uranusMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_uranusMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("fieldstone.ppm", true)));
    g_uranusMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("uranus.ppm", false)));
    
    // pluto material
    g_plutoMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_plutoMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("pluto.ppm", true)));
    g_plutoMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("pluto.ppm", false)));
    
    // asteroid material
    g_asteroidMat.reset(new Material("./shaders/normal-gl3.vshader",
                 "./shaders/normal-gl3.fshader"));
    g_asteroidMat->getUniforms().put("uTexColor",shared_ptr<ImageTexture>(new ImageTexture("asteroid.ppm", true)));
    g_asteroidMat->getUniforms().put("uTexNormal", shared_ptr<ImageTexture>(new ImageTexture("asteroid.ppm", false)));
    
    // copy solid prototype, and set to wireframed rendering
    g_arcballMat.reset(new Material(solid));
    g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
    g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // copy solid prototype, and set to color white
    g_lightMat.reset(new Material(solid));
    
    // LUIS: MADE BLACK FOR NOW, NEED TO MAKE TRANSPARENT
    g_lightMat->getUniforms().put("uColor", Cvec3f(255., 255., 255.0));
    // pick shader
    g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader",
                                    "./shaders/pick-gl3.fshader"));
};
static void initGeometry() {
//    initGround();
    initCubes();
    initSphere();
}
static float getRand();
static void constructCelestial(shared_ptr<SgTransformNode> base,
                           shared_ptr<Material> material) {
    // fix these later, not accurate
    const float SUN_RADIUS = 1.0, MERCURY = 1./38.5, VENUS = 1./23., EARTH = 1./24., MARS = 1./30, JUPITER = 1./8., SATURN = 1./10., URANUS = 1./20.4, NEPTUNE = 1./20.7, PLUTO = 1./50. ; // #giveplutorights
    const int NUM_JOINTS = 10, NUM_SHAPES = 10;
    struct JointDesc {
        int parent;
        float x, y, z;
    };
    JointDesc jointDesc[NUM_JOINTS] = {
        {-1},       // SUN
        {0, 0, 0, 0},  // MERCURY
        {0, 0, 0, 0},  // VENUS
        {0, 0, 0, 0},    // EARTH
        {0, 0, 0, 0},  // MARS
        {0, 0, 0, 0},  // JUPITER
        {0, 0, 0, 0},  // SATURN
        {0, 0, 0, 0},  // URANUS
        {0, 0, 0, 0},  // NEPTURE
        {0, 0, 0, 0}   // PLUTO
    };
    struct ShapeDesc {
        int parentJointId;
        float x, y, z, sx, sy, sz;
        shared_ptr<Geometry> geometry;
        shared_ptr<Material> material;
    };
    
//    ShapeDesc shapeDesc[NUM_PLANETS+1];
    
    ShapeDesc shapeDesc[NUM_PLANETS + 1] = {
         {0, 0, 0, 0, SUN_RADIUS, SUN_RADIUS, SUN_RADIUS, g_sphere,
          g_sunMat}, // SUN
         {1, 1.3, 0, 0, SUN_RADIUS * MERCURY , SUN_RADIUS * MERCURY, SUN_RADIUS * MERCURY,
          g_sphere, g_mercMat}, // MERCURY
         {2, 1.6, 0, 0, SUN_RADIUS * VENUS, SUN_RADIUS * VENUS, SUN_RADIUS * VENUS,
          g_sphere, g_venusMat}, // VENUS
         {3, 1.9, 0, 0, SUN_RADIUS * EARTH, SUN_RADIUS * EARTH, SUN_RADIUS * EARTH, g_sphere,
          g_earthMat}, // EARTH
         {4, 2.2, 0, 0, SUN_RADIUS * MARS, SUN_RADIUS * MARS, SUN_RADIUS * MARS, g_sphere,
          g_marsMat}, // MARS
         {5, 2.5, 0, 0, SUN_RADIUS * JUPITER, SUN_RADIUS * JUPITER, SUN_RADIUS * JUPITER,
          g_sphere, g_jupiterMat}, // JUPITER
         {6, 2.8, 0, 0, SUN_RADIUS * SATURN, SUN_RADIUS * SATURN, SUN_RADIUS * SATURN,
          g_sphere, g_saturnMat}, // SATURN
         {7, 3.1, 0, 0, SUN_RADIUS * URANUS, SUN_RADIUS * URANUS, SUN_RADIUS * URANUS, g_sphere,
          g_uranusMat}, // URANUS
         {8, 3.4, 0, 0, SUN_RADIUS * NEPTUNE, SUN_RADIUS * NEPTUNE, SUN_RADIUS * NEPTUNE, g_sphere,
          g_neptuneMat}, // NEPTUNE
         {9, 3.7, 0, 0, SUN_RADIUS * PLUTO, SUN_RADIUS * PLUTO, SUN_RADIUS * PLUTO, g_sphere, g_plutoMat}, //PLUTO
     };
    
    if (!toScale){
        for (int i = 1; i < NUM_PLANETS+1; i++){
            float dist = shapeDesc[i].x;
            float theta = getRand() * 3.14159 * 2;
            if (inLine) theta = 0;
            float phi = planetData[i-1].inclination * 2 * 3.14159 / 360.;
            float x = dist * cos(theta) * cos(phi);
            float z = dist * sin(theta) * cos(phi);
            float y = dist * sin(phi);
            float rad = shapeDesc[i].sx;
            shapeDesc[i] = {i, x, y, z, rad,rad,rad, g_sphere, shapeDesc[i].material};
            
            planetData[i-1].theta_at_peak = theta;
            
//            cerr << "Planet number: " << i << " has distance: " << dist << "\n \t\tand radius " << rad << "\n";
        }
    }
    if (toScale){
        for (int i = 1; i < NUM_PLANETS+1; i++){
            float dist = shapeDesc[i].x;
            float theta = getRand() * 3.14159 * 2;
            if (inLine) theta = 0;
            float phi = planetData[i-1].inclination * 2 * 3.14159 / 360.;
            float x = dist * cos(theta) * cos(phi);
            float z = dist * sin(theta) * cos(phi);
            float y = dist * sin(phi);
    
            float rad = (planetData[i-1].diameter)/1.39e6;
            shapeDesc[i] = {i, x, y, z, rad,rad,rad, g_sphere, shapeDesc[i].material};
            planetData[i-1].theta_at_peak = theta;
            
//            cerr << "Planet number: " << i << " has distance: " << dist << "\n \t\tand radius " << rad << "\n";
        }
    }
    
    for (int i = 0; i < NUM_JOINTS; ++i) {
        if (jointDesc[i].parent == -1)
            jointNodes[i] = base;
        else {
            jointNodes[i].reset(new SgRbtNode(RigTForm(
                Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
            jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
        }
    }
    for (int i = 0; i < NUM_SHAPES; ++i) {
        shared_ptr<MyShapeNode> shape(new MyShapeNode(
            shapeDesc[i].geometry, shapeDesc[i].material,
            Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
            Cvec3(90, 0, 0), // make this 90 to fix materials
            Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
        jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
    }
}
static void updatePlanets(){
    for (int i = 1; i < NUM_PLANETS + 1; i++){
        float alpha = planetData[i-1].theta_at_peak;
        float phi = planetData[i-1].inclination * 2 * 3.14159 / 360.;
        // Account for inclination and such, should be normal to plane of orbit
        float x = -sin(phi) * cos(alpha);
        float y = cos(phi);
        float z = -sin(phi) * sin(alpha);
        Cvec3 k = Cvec3(x,y,z);
        float theta = (1/planetData[i-1].period)*2 * 3.14159 * timeRatio;
        theta = theta/2;
        RigTForm q = RigTForm(Quat(cos(theta), k * sin(theta)));
        shared_ptr<SgRbtNode> sgRbt = dynamic_pointer_cast<SgRbtNode>(jointNodes[i]);
        assert(sgRbt != NULL);
        RigTForm newRbt = sgRbt->getRbt() * q;
        sgRbt->setRbt(newRbt);
    }
}
static float getRand(){
    int x = rand() % 1000;
    float fx = (float) x;
    return fx/1000.0;
}
static void constructStars(shared_ptr<SgTransformNode> base,
                           shared_ptr<Material> material) {
    
    const float star_radius = .1;
    const int NUM_STARS = 600;
    
    struct ShapeDesc {
        float x, y, z;
        float sx, sy, sz;
        shared_ptr<Geometry> geometry;
        shared_ptr<Material> material;
    };
    
    ShapeDesc shapeDesc[NUM_STARS];
    
    for (int i = 0; i < NUM_STARS; ++i) {
        float r =  getRand() + 30.;
        float theta = getRand() * 2. * 3.14159;
        float phi = getRand() * 3.14159;
        float x = cos(theta) * sin(phi) * r;
        float y = sin(theta) * sin(phi) * r;
        float z = cos(phi) * r;
        shapeDesc[i] =  {x, y, z, star_radius, star_radius, star_radius, g_sphere,
            material}; // SUN
    }
    shared_ptr<SgTransformNode> jointNodes[NUM_STARS];
    
    for (int i = 0; i < NUM_STARS; ++i) {
        shared_ptr<MyShapeNode> shape(new MyShapeNode(
            shapeDesc[i].geometry, shapeDesc[i].material,
            Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
            Cvec3(0, 0, 0),
            Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
        g_world->addChild(shape);
    }
}
static void constructAsteroidBelt(shared_ptr<SgTransformNode> base,
                           shared_ptr<Material> material) {
    const float asteroid_radius = .01;
    const int NUM_ASTEROIDS = 200;
    struct ShapeDesc {
        float x, y, z;
        float sx, sy, sz;
        shared_ptr<Geometry> geometry;
        shared_ptr<Material> material;
    };
    ShapeDesc shapeDesc[NUM_ASTEROIDS];
    
    for (int i = 0; i < NUM_ASTEROIDS; ++i) {
        float r =  2.3 + distribution(generator)*.03;
        float theta = getRand() * 2. * 3.14159;
        float x = cos(theta) * r;
        float y = .01 + distribution(generator)*.03;
        float z = sin(theta) * r;
    
        
        shapeDesc[i] =  {x, y, z, asteroid_radius, asteroid_radius, asteroid_radius, g_sphere,
            g_asteroidMat}; // SUN
    }
    shared_ptr<SgTransformNode> jointNodes[NUM_ASTEROIDS];
    
    for (int i = 0; i < NUM_ASTEROIDS; ++i) {
        shared_ptr<MyShapeNode> shape(new MyShapeNode(
            shapeDesc[i].geometry, shapeDesc[i].material,
            Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
            Cvec3(0, 0, 0),
            Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
        g_world->addChild(shape);
    }
}
static void constructKuiperBelt(shared_ptr<SgTransformNode> base,
                           shared_ptr<Material> material) {
    const float asteroid_radius = .007;
    const int NUM_ASTEROIDS = 800;
    struct ShapeDesc {
        float x, y, z;
        float sx, sy, sz;
        shared_ptr<Geometry> geometry;
        shared_ptr<Material> material;
    };
    ShapeDesc shapeDesc[NUM_ASTEROIDS];
    
    for (int i = 0; i < NUM_ASTEROIDS; ++i) {
        float r =  getRand() * (5. - 3.9) + 3.9;
        r = 4.5 + distribution(generator) * .3;
        float theta = getRand() * 2. * 3.14159;
        float x = cos(theta) * r;
        float y = .01 + distribution(generator)*.13;
        float z = sin(theta) * r;
        
        float distanceFromCenter = sqrt(x * x + y * y + z * z);
        
        shapeDesc[i] =  {x, y, z, asteroid_radius, asteroid_radius, asteroid_radius, g_sphere,
            g_asteroidMat}; // SUN
    }
    shared_ptr<SgTransformNode> jointNodes[NUM_ASTEROIDS];
    
    for (int i = 0; i < NUM_ASTEROIDS; ++i) {
        shared_ptr<MyShapeNode> shape(new MyShapeNode(
            shapeDesc[i].geometry, shapeDesc[i].material,
            Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
            Cvec3(0, 0, 0),
            Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
        g_world->addChild(shape);
    }
}
static void initScene() {
    g_world.reset(new SgRootNode());
    g_skyNode.reset(new SgRbtNode(initSkyRbt));
    
    
    
    g_solarSystem.reset(new SgRbtNode(RigTForm(Cvec3(0, 0, 0))));
    constructCelestial(g_solarSystem, g_planetMat);  // a Red robot
    if (!toScale){
        g_stars.reset(new SgRbtNode(RigTForm(Cvec3(0, 0, 0))));
        constructStars(g_stars, g_lightMat);
        
        g_asteroidBelt.reset(new SgRbtNode(RigTForm(Cvec3(0, 0, 0))));
        constructAsteroidBelt(g_asteroidBelt, g_astMat);
        
        g_kuiperBelt.reset(new SgRbtNode(RigTForm(Cvec3(0, 0, 0))));
        constructKuiperBelt(g_kuiperBelt, g_astMat);
    }
    
    
    
    g_light1.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.0, 0.0))));
    g_light1->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_sphere, g_lightMat, Cvec3(0), Cvec3(0), Cvec3(0.5))));
    g_world->addChild(g_skyNode);
    g_world->addChild(g_solarSystem);
    g_world->addChild(g_light1);
    g_world->addChild(g_stars);
    
    g_world->addChild(g_asteroidBelt);
    
    g_world->addChild(g_kuiperBelt);
    
    g_currentCameraNode = g_skyNode;
    
}
static void glfwLoop() {
    g_lastFrameClock = glfwGetTime();
    while (!glfwWindowShouldClose(g_window)) {
        double thisTime = glfwGetTime();
        if( thisTime - g_lastFrameClock >= 1. / g_framesPerSecond) {
            animationUpdate();
            display();
            g_lastFrameClock = thisTime;
        }
        glfwPollEvents();
    }
}
int main(int argc, char *argv[]) {
    try {
        initGlfwState();
        // on Mac, we shouldn't use GLEW.
#ifndef __MAC__
        glewInit(); // load the OpenGL extensions
#endif
        initGLState();
        initMaterials();
        initGeometry();
        initScene();
        glfwLoop();
        return 0;
    } catch (const runtime_error &e) {
        cout << "Exception caught: " << e.what() << endl;
        return -1;
    }
}

