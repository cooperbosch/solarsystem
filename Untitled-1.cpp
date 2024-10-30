////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////

#include <cstddef>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
#include "mesh.h"

using namespace std;

// G L O B A L S ///////////////////////////////////////////////////

static bool g_shellNeedsUpdate = false;


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

static bool g_playingAnimation = false;

// --------- Materials
static shared_ptr<Material> g_redDiffuseMat, g_blueDiffuseMat, g_bumpFloorMat,
    g_arcballMat, g_pickingMat, g_lightMat;

shared_ptr<Material> g_overridingMaterial;

// --------- Geometry
typedef SgGeometryShapeNode MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;

// --------- Scene

static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node,
    g_robot2Node, g_light1, g_light2;

static shared_ptr<SgRbtNode> g_currentCameraNode;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode;

// ---------- Animation

class Animator {
  public:
    typedef vector<shared_ptr<SgRbtNode>> SgRbtNodes;
    typedef vector<RigTForm> KeyFrame;
    typedef list<KeyFrame> KeyFrames;
    typedef KeyFrames::iterator KeyFrameIter;

  private:
    SgRbtNodes nodes_;
    KeyFrames keyFrames_;

  public:
    void attachSceneGraph(shared_ptr<SgNode> root) {
        nodes_.clear();
        keyFrames_.clear();
        dumpSgRbtNodes(root, nodes_);
    }

    void loadAnimation(const char *filename) {
        ifstream f(filename, ios::binary);
        if (!f)
            throw runtime_error(string("Cannot load ") + filename);
        int numFrames, numRbtsPerFrame;
        f >> numFrames >> numRbtsPerFrame;
        if (numRbtsPerFrame != nodes_.size()) {
            cerr << "Number of Rbt per frame in " << filename
                 << " does not match number of SgRbtNodes in the current scene "
                    "graph.";
            return;
        }

        Cvec3 t;
        Quat r;
        keyFrames_.clear();
        for (int i = 0; i < numFrames; ++i) {
            keyFrames_.push_back(KeyFrame());
            keyFrames_.back().reserve(numRbtsPerFrame);
            for (int j = 0; j < numRbtsPerFrame; ++j) {
                f >> t[0] >> t[1] >> t[2] >> r[0] >> r[1] >> r[2] >> r[3];
                keyFrames_.back().push_back(RigTForm(t, r));
            }
        }
    }

    void saveAnimation(const char *filename) {
        ofstream f(filename, ios::binary);
        int numRbtsPerFrame = nodes_.size();
        f << getNumKeyFrames() << ' ' << numRbtsPerFrame << '\n';
        for (KeyFrames::const_iterator frameIter = keyFrames_.begin(),
                                       e = keyFrames_.end();
             frameIter != e; ++frameIter) {
            for (int j = 0; j < numRbtsPerFrame; ++j) {
                const RigTForm &rbt = (*frameIter)[j];
                const Cvec3 &t = rbt.getTranslation();
                const Quat &r = rbt.getRotation();
                f << t[0] << ' ' << t[1] << ' ' << t[2] << ' ' << r[0] << ' '
                  << r[1] << ' ' << r[2] << ' ' << r[3] << '\n';
            }
        }
    }

    int getNumKeyFrames() const { return keyFrames_.size(); }

    int getNumRbtNodes() const { return nodes_.size(); }

    // t can be in the range [0, keyFrames_.size()-3]. Fractional amount
    // like 1.5 is allowed.
    void animate(double t) {
        if (t < 0 || t > keyFrames_.size() - 3)
            throw runtime_error("Invalid animation time parameter. Must be in "
                                "the range [0, numKeyFrames - 3]");

        t += 1; // interpret the key frames to be at t= -1, 0, 1, 2, ...
        const int integralT = int(floor(t));
        const double fraction = t - integralT;

        KeyFrameIter f1 = getNthKeyFrame(integralT), f0 = f1, f2 = f1;
        --f0;
        ++f2;
        KeyFrameIter f3 = f2;
        ++f3;
        if (f3 == keyFrames_.end()) // this might be true when t is exactly
                                    // keyFrames_.size()-3.
            f3 = f2;                // in which case we step back

        for (int i = 0, n = nodes_.size(); i < n; ++i) {
            nodes_[i]->setRbt(interpolateCatmullRom(
                (*f0)[i], (*f1)[i], (*f2)[i], (*f3)[i], fraction));
        }
    }

    KeyFrameIter keyFramesBegin() { return keyFrames_.begin(); }

    KeyFrameIter keyFramesEnd() { return keyFrames_.end(); }

    KeyFrameIter getNthKeyFrame(int n) {
        KeyFrameIter frameIter = keyFrames_.begin();
        advance(frameIter, n);
        return frameIter;
    }

    void deleteKeyFrame(KeyFrameIter keyFrameIter) {
        keyFrames_.erase(keyFrameIter);
    }

    void pullKeyFrameFromSg(KeyFrameIter keyFrameIter) {
        for (int i = 0, n = nodes_.size(); i < n; ++i) {
            (*keyFrameIter)[i] = nodes_[i]->getRbt();
        }
    }

    void pushKeyFrameToSg(KeyFrameIter keyFrameIter) {
        for (int i = 0, n = nodes_.size(); i < n; ++i) {
            nodes_[i]->setRbt((*keyFrameIter)[i]);
        }
    }

    KeyFrameIter insertEmptyKeyFrameAfter(KeyFrameIter beforeFrame) {
        if (beforeFrame != keyFrames_.end())
            ++beforeFrame;

        KeyFrameIter frameIter = keyFrames_.insert(beforeFrame, KeyFrame());
        frameIter->resize(nodes_.size());
        return frameIter;
    }
};
static int g_msBetweenKeyFrames = 2000; // 2 seconds between keyframes
static int g_framesPerSecond = 60; // frames to render per second during animation playback

//static double g_lastFrameClock;
static int g_animateTime = 0;

static Animator g_animator;
static Animator::KeyFrameIter g_curKeyFrame;
static int g_curKeyFrameNum;

// BUNNY GLOBALS

// Material
static shared_ptr<Material> g_bunnyMat; // for the bunny

static vector<shared_ptr<Material>> g_bunnyShellMats; // for bunny shells

// New Geometry
static const int g_numShells = 24; // constants defining how many layers of shells
static double g_furHeight = 0.21;
static double g_hairyness = 0.7;

static shared_ptr<SimpleGeometryPN> g_bunnyGeometry;
static vector<shared_ptr<SimpleGeometryPNX>> g_bunnyShellGeometries;
static Mesh g_bunnyMesh;

// New Scene node
static shared_ptr<SgRbtNode> g_bunnyNode;

// For Simulation
static double g_lastFrameClock;

// Global variables for used physical simulation
static const Cvec3 g_gravity(0, -0.5, 0); // gavity vector
static double g_timeStep = 0.02;
static double g_numStepsPerFrame = 5;
static double g_damping = 0.96;
static double g_stiffness = 4;
static int g_simulationsPerSecond = 60;

static std::vector<Cvec3>
    g_tipPos,      // should be hair tip pos in world-space coordinates
    g_tipVelocity; // should be hair tip velocity in world-space coordinates


///////////////// END OF G L O B A L S /////////////////////////////////////////

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

static void initRobots() {
    // Init whatever extra geometry  needed for the robots
}

// takes a projection matrix and send to the the shaders
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

static void updateShellGeometry();
static void hairsSimulationUpdate();

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
    Cvec3 l2 = getPathAccumRbt(g_world, g_light2).getTranslation();
    uniforms.put("uLight", Cvec3(invEyeRbt * Cvec4(l1, 1)));
    uniforms.put("uLight2", Cvec3(invEyeRbt * Cvec4(l2, 1)));

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
    updateShellGeometry();
//    hairsSimulationUpdate();
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

bool interpolateAndDisplay(float t) {
    if (t > g_animator.getNumKeyFrames() - 3)
        return true;
    g_animator.animate(t);
    return false;
}

static void animationUpdate() {
    if (g_playingAnimation) {
        bool endReached = interpolateAndDisplay((float) g_animateTime / g_msBetweenKeyFrames);
        if (!endReached)
            g_animateTime += 1000./g_framesPerSecond;
        else {
            cerr << "Finished playing animation" << endl;
            g_curKeyFrame = g_animator.keyFramesEnd();
            advance(g_curKeyFrame, -2);
            g_animator.pushKeyFrameToSg(g_curKeyFrame);
            g_playingAnimation = false;
            g_animateTime = 0;
            g_curKeyFrameNum = g_animator.getNumKeyFrames() - 2;
            cerr << "Now at frame [" << g_curKeyFrameNum << "]" << endl;
        }
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
                 << "s\t\tsave screenshot\n"
                 << "f\t\tToggle flat shading on/off.\n"
                 << "p\t\tUse mouse to pick a part to edit\n"
                 << "v\t\tCycle view\n"
                 << "drag left mouse to rotate\n"
                 << "a\t\tToggle display arcball\n"
                 << "w\t\tWrite animation to animation.txt\n"
                 << "i\t\tRead animation from animation.txt\n"
                 << "c\t\tCopy frame to scene\n"
                 << "u\t\tCopy sceneto frame\n"
                 << "n\t\tCreate new frame after current frame and copy scene to "
                "it\n"
                 << "d\t\tDelete frame\n"
                 << ">\t\tGo to next frame\n"
                 << "<\t\tGo to prev. frame\n"
                 << "y\t\tPlay/Stop animation\n"
                 << endl;
            break;
        case GLFW_KEY_S:
            glFlush();
            writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
            break;
        case GLFW_KEY_V: {
            shared_ptr<SgRbtNode> viewers[] = {g_skyNode, g_robot1Node,
                                               g_robot2Node};
            for (int i = 0; i < 3; ++i) {
                if (g_currentCameraNode == viewers[i]) {
                    g_currentCameraNode = viewers[(i + 1) % 3];
                    break;
                }
            }
        } break;
        case GLFW_KEY_P:
            g_pickingMode = !g_pickingMode;
            cerr << "Picking mode is " << (g_pickingMode ? "on" : "off") << endl;
            break;
        case GLFW_KEY_M:
            g_activeCameraFrame = SkyMode((g_activeCameraFrame + 1) % 2);
            cerr << "Editing sky eye w.r.t. "
                 << (g_activeCameraFrame == WORLD_SKY ? "world-sky frame\n"
                     : "sky-sky frame\n")
                 << endl;
            break;
        case GLFW_KEY_A:
            g_displayArcball = !g_displayArcball;
            break;
        case GLFW_KEY_U:
            if (g_playingAnimation) {
                cerr << "Cannot operate when playing animation" << endl;
                break;
            }

            if (g_curKeyFrame ==
                g_animator
                .keyFramesEnd()) { // only possible when frame list is empty
                cerr << "Create new frame [0]." << endl;
                g_curKeyFrame = g_animator.insertEmptyKeyFrameAfter(
                                                                    g_animator.keyFramesBegin());
                g_curKeyFrameNum = 0;
            }
            cerr << "Copying scene graph to current frame [" << g_curKeyFrameNum
                 << "]" << endl;
            g_animator.pullKeyFrameFromSg(g_curKeyFrame);
            break;
        case GLFW_KEY_N:
            if (g_playingAnimation) {
                cerr << "Cannot operate when playing animation" << endl;
                break;
            }
            if (g_animator.getNumKeyFrames() != 0)
                ++g_curKeyFrameNum;
            g_curKeyFrame = g_animator.insertEmptyKeyFrameAfter(g_curKeyFrame);
            g_animator.pullKeyFrameFromSg(g_curKeyFrame);
            cerr << "Create new frame [" << g_curKeyFrameNum << "]" << endl;
            break;
        case GLFW_KEY_C:
            if (g_playingAnimation) {
                cerr << "Cannot operate when playing animation" << endl;
                break;
            }
            if (g_curKeyFrame != g_animator.keyFramesEnd()) {
                cerr << "Loading current key frame [" << g_curKeyFrameNum
                     << "] to scene graph" << endl;
                g_animator.pushKeyFrameToSg(g_curKeyFrame);
            } else {
                cerr << "No key frame defined" << endl;
            }
            break;
        case GLFW_KEY_D:
            if (g_playingAnimation) {
                cerr << "Cannot operate when playing animation" << endl;
                break;
            }
            if (g_curKeyFrame != g_animator.keyFramesEnd()) {
                Animator::KeyFrameIter newCurKeyFrame = g_curKeyFrame;
                cerr << "Deleting current frame [" << g_curKeyFrameNum << "]"
                     << endl;
                ;
                if (g_curKeyFrame == g_animator.keyFramesBegin()) {
                    ++newCurKeyFrame;
                } else {
                    --newCurKeyFrame;
                    --g_curKeyFrameNum;
                }
                g_animator.deleteKeyFrame(g_curKeyFrame);
                g_curKeyFrame = newCurKeyFrame;
                if (g_curKeyFrame != g_animator.keyFramesEnd()) {
                    g_animator.pushKeyFrameToSg(g_curKeyFrame);
                    cerr << "Now at frame [" << g_curKeyFrameNum << "]" << endl;
                } else
                    cerr << "No frames defined" << endl;
            } else {
                cerr << "Frame list is now EMPTY" << endl;
            }
            break;
        case GLFW_KEY_PERIOD: // >
            if (!(mods & GLFW_MOD_SHIFT)) break;
            if (g_playingAnimation) {
                cerr << "Cannot operate when playing animation" << endl;
                break;
            }
            if (g_curKeyFrame != g_animator.keyFramesEnd()) {
                if (++g_curKeyFrame == g_animator.keyFramesEnd())
                    --g_curKeyFrame;
                else {
                    ++g_curKeyFrameNum;
                    g_animator.pushKeyFrameToSg(g_curKeyFrame);
                    cerr << "Stepped forward to frame [" << g_curKeyFrameNum << "]"
                         << endl;
                }
            }
            break;
        case GLFW_KEY_COMMA: // <
            if (!(mods & GLFW_MOD_SHIFT)) break;
            if (g_playingAnimation) {
                cerr << "Cannot operate when playing animation" << endl;
                break;
            }
            if (g_curKeyFrame != g_animator.keyFramesBegin()) {
                --g_curKeyFrame;
                --g_curKeyFrameNum;
                g_animator.pushKeyFrameToSg(g_curKeyFrame);
                cerr << "Stepped backward to frame [" << g_curKeyFrameNum << "]"
                     << endl;
            }
            break;
        case GLFW_KEY_W:
            cerr << "Writing animation to animation.txt\n";
            g_animator.saveAnimation("animation.txt");
            break;
        case GLFW_KEY_I:
            if (g_playingAnimation) {
                cerr << "Cannot operate when playing animation" << endl;
                break;
            }
            cerr << "Reading animation from animation.txt\n";
            g_animator.loadAnimation("animation.txt");
            g_curKeyFrame = g_animator.keyFramesBegin();
            cerr << g_animator.getNumKeyFrames() << " frames read.\n";
            if (g_curKeyFrame != g_animator.keyFramesEnd()) {
                g_animator.pushKeyFrameToSg(g_curKeyFrame);
                cerr << "Now at frame [0]" << endl;
            }
            g_curKeyFrameNum = 0;
            break;
        case GLFW_KEY_MINUS:
            g_msBetweenKeyFrames = min(g_msBetweenKeyFrames + 100, 10000);
            cerr << g_msBetweenKeyFrames << " ms between keyframes.\n";
            break;
        case GLFW_KEY_EQUAL: // +
            if (!(mods & GLFW_MOD_SHIFT)) break;
            g_msBetweenKeyFrames = max(g_msBetweenKeyFrames - 100, 100);
            cerr << g_msBetweenKeyFrames << " ms between keyframes.\n";
            break;
        case GLFW_KEY_Y:
            if (!g_playingAnimation) {
                if (g_animator.getNumKeyFrames() < 4) {
                    cerr << " Cannot play animation with less than 4 keyframes."
                         << endl;
                } else {
                    g_playingAnimation = true;
                    cerr << "Playing animation... " << endl;
                }
            } else {
                cerr << "Stopping animation... " << endl;
                g_playingAnimation = false;
            }
            break;
        case GLFW_KEY_RIGHT:
            g_furHeight *= 1.05;
            cerr << "fur height = " << g_furHeight << std::endl;
            break;
        case GLFW_KEY_LEFT:
            g_furHeight /= 1.05;
            std::cerr << "fur height = " << g_furHeight << std::endl;
            break;
        case GLFW_KEY_UP:
            g_hairyness *= 1.05;
            cerr << "hairyness = " << g_hairyness << std::endl;
            break;
        case GLFW_KEY_DOWN:
            g_hairyness /= 1.05;
            cerr << "hairyness = " << g_hairyness << std::endl;
            break;
        case GLFW_KEY_Q:
            cerr << "updating hair\n";
            hairsSimulationUpdate();
            break;
        }
    } else {
        switch(key) {
        case GLFW_KEY_SPACE:
            g_spaceDown = false;
            break;
        }
    }

    // Sanity check that our g_curKeyFrameNum is in sync with the g_curKeyFrame
    if (g_animator.getNumKeyFrames() > 0)
        assert(g_animator.getNthKeyFrame(g_curKeyFrameNum) == g_curKeyFrame);
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
    glClearColor(128. / 255., 200. / 255., 255. / 255., 0.);
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

// BUNNY CODE
// New function that loads the bunny mesh and initializes the bunny shell meshes
static void initBunnyMeshes() {
    g_bunnyMesh.load("bunny.mesh");

    Cvec3 sumFaces = Cvec3();
    int ctr = 0;

    // TODO: Init the per vertex normal of g_bunnyMesh; see "calculating
    // normals" section of spec
    for (int i = 0; i < g_bunnyMesh.getNumVertices(); ++i) {
        const Mesh::Vertex v = g_bunnyMesh.getVertex(i);
        
        Mesh::VertexIterator it(v.getIterator()), it0(it);
        do
        {
            sumFaces += it.getFace().getNormal();
            ctr++;
            // can use here it.getVertex(), it.getFace()
        }
        while (++it != it0);
        // go around once the 1ring
        v.setNormal(sumFaces/ctr);
        
        // ResetCounter and yourMum
        ctr = 0;
        sumFaces= Cvec3();
        
    }
    

    // TODO: Initialize g_bunnyGeometry from g_bunnyMesh; see "mesh preparation"
    // section of spec
    // ...

    vector<VertexPN> meshVertices;
    
    meshVertices.resize(g_bunnyMesh.getNumFaces() * 3);
    
    // Initialize g_bunnyGeometry from g_bunnyMesh
    for (int i = 0; i < g_bunnyMesh.getNumFaces(); ++i) {
        // TODO: ADD TEXTURE LATER
        for (int j = 0; j < 3; j++){
            Mesh::Vertex v = g_bunnyMesh.getFace(i).getVertex(j);
            meshVertices[3*i + j] = VertexPN(v.getPosition(), v.getNormal());
        }
    }
    
    g_bunnyGeometry.reset(new SimpleGeometryPN);
    
    g_bunnyGeometry->upload(&meshVertices[0], g_bunnyMesh.getNumFaces() * 3);

    // Now allocate array of SimpleGeometryPNX to for shells, one per layer
    g_bunnyShellGeometries.resize(g_numShells);
    for (int i = 0; i < g_numShells; ++i) {
        g_bunnyShellGeometries[i].reset(new SimpleGeometryPNX());
    }
}


static void initMaterials() {
    // Create some prototype materials
    Material diffuse("./shaders/basic-gl3.vshader",
                     "./shaders/diffuse-gl3.fshader");
    Material solid("./shaders/basic-gl3.vshader",
                   "./shaders/solid-gl3.fshader");

    // copy diffuse prototype and set red color
    g_redDiffuseMat.reset(new Material(diffuse));
    g_redDiffuseMat->getUniforms().put("uColor", Cvec3f(1, 0, 0));

    // copy diffuse prototype and set blue color
    g_blueDiffuseMat.reset(new Material(diffuse));
    g_blueDiffuseMat->getUniforms().put("uColor", Cvec3f(0, 0, 1));

    // normal mapping
    g_bumpFloorMat.reset(new Material("./shaders/normal-gl3.vshader",
                                      "./shaders/normal-gl3.fshader"));
    g_bumpFloorMat->getUniforms().put(
        "uTexColor",
        shared_ptr<ImageTexture>(new ImageTexture("Fieldstone.ppm", true)));
    g_bumpFloorMat->getUniforms().put(
        "uTexNormal", shared_ptr<ImageTexture>(
                          new ImageTexture("FieldstoneNormal.ppm", false)));

    // copy solid prototype, and set to wireframed rendering
    g_arcballMat.reset(new Material(solid));
    g_arcballMat->getUniforms().put("uColor", Cvec3f(0.27f, 0.82f, 0.35f));
    g_arcballMat->getRenderStates().polygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // copy solid prototype, and set to color white
    g_lightMat.reset(new Material(solid));
    g_lightMat->getUniforms().put("uColor", Cvec3f(1, 1, 1));

    // pick shader
    g_pickingMat.reset(new Material("./shaders/basic-gl3.vshader",
                                    "./shaders/pick-gl3.fshader"));
    
    // bunny material
    g_bunnyMat.reset(new Material("./shaders/basic-gl3.vshader",
                                  "./shaders/bunny-gl3.fshader"));
    g_bunnyMat->getUniforms().put("uColorAmbient", Cvec3f(0.45f, 0.3f, 0.3f))
                             .put("uColorDiffuse", Cvec3f(0.2f, 0.2f, 0.2f));

    // bunny shell materials;
    // common shell texture:
    shared_ptr<ImageTexture> shellTexture(new ImageTexture("shell.ppm", false));

    // enable repeating of texture coordinates
    shellTexture->bind();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // eachy layer of the shell uses a different material, though the materials
    // will share the same shader files and some common uniforms. hence we
    // create a prototype here, and will copy from the prototype later
    Material bunnyShellMatPrototype("./shaders/bunny-shell-gl3.vshader",
                                    "./shaders/bunny-shell-gl3.fshader");
    bunnyShellMatPrototype.getUniforms().put("uTexShell", shellTexture);
    bunnyShellMatPrototype.getRenderStates()
        .blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) // set blending mode
        .enable(GL_BLEND)                                // enable blending
        .disable(GL_CULL_FACE);                          // disable culling

    // allocate array of materials
    g_bunnyShellMats.resize(g_numShells);
    for (int i = 0; i < g_numShells; ++i) {
        // copy prototype
        g_bunnyShellMats[i].reset(new Material(bunnyShellMatPrototype));
        // but set a different exponent for blending transparency
        g_bunnyShellMats[i]->getUniforms().put(
            "uAlphaExponent", 2.f + 5.f * float(i + 1) / g_numShells);
    }
};

static void initGeometry() {
    initGround();
    initCubes();
    initSphere();
    initRobots();
    initBunnyMeshes();
}

static void constructRobot(shared_ptr<SgTransformNode> base,
                           shared_ptr<Material> material) {
    const float ARM_LEN = 0.7, ARM_THICK = 0.25, LEG_LEN = 1, LEG_THICK = 0.25,
                 TORSO_LEN = 1.5, TORSO_THICK = 0.25, TORSO_WIDTH = 1,
                 HEAD_SIZE = 0.7;
    const int NUM_JOINTS = 10, NUM_SHAPES = 10;

    struct JointDesc {
        int parent;
        float x, y, z;
    };

    JointDesc jointDesc[NUM_JOINTS] = {
        {-1},                                    // torso
        {0, TORSO_WIDTH / 2, TORSO_LEN / 2, 0},  // upper right arm
        {0, -TORSO_WIDTH / 2, TORSO_LEN / 2, 0}, // upper left arm
        {1, ARM_LEN, 0, 0},                      // lower right arm
        {2, -ARM_LEN, 0, 0},                     // lower left arm
        {0, TORSO_WIDTH / 2 - LEG_THICK / 2, -TORSO_LEN / 2,
         0}, // upper right leg
        {0, -TORSO_WIDTH / 2 + LEG_THICK / 2, -TORSO_LEN / 2,
         0},                     // upper left leg
        {5, 0, -LEG_LEN, 0},     // lower right leg
        {6, 0, -LEG_LEN, 0},     // lower left
        {0, 0, TORSO_LEN / 2, 0} // head
    };

    struct ShapeDesc {
        int parentJointId;
        float x, y, z, sx, sy, sz;
        shared_ptr<Geometry> geometry;
        shared_ptr<Material> material;
    };

    ShapeDesc shapeDesc[NUM_SHAPES] = {
        {0, 0, 0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube,
         material}, // torso
        {1, ARM_LEN / 2, 0, 0, ARM_LEN / 2, ARM_THICK / 2, ARM_THICK / 2,
         g_sphere, material}, // upper right arm
        {2, -ARM_LEN / 2, 0, 0, ARM_LEN / 2, ARM_THICK / 2, ARM_THICK / 2,
         g_sphere, material}, // upper left arm
        {3, ARM_LEN / 2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube,
         material}, // lower right arm
        {4, -ARM_LEN / 2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube,
         material}, // lower left arm
        {5, 0, -LEG_LEN / 2, 0, LEG_THICK / 2, LEG_LEN / 2, LEG_THICK / 2,
         g_sphere, material}, // upper right leg
        {6, 0, -LEG_LEN / 2, 0, LEG_THICK / 2, LEG_LEN / 2, LEG_THICK / 2,
         g_sphere, material}, // upper left leg
        {7, 0, -LEG_LEN / 2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube,
         material}, // lower right leg
        {8, 0, -LEG_LEN / 2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube,
         material}, // lower left leg
        {9, 0, HEAD_SIZE / 2 * 1.5f, 0, HEAD_SIZE / 2, HEAD_SIZE / 2,
         HEAD_SIZE / 2, g_sphere, material}, // head
    };

    shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

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
            Cvec3(0, 0, 0),
            Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
        jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
    }
}

static void initScene() {
    g_world.reset(new SgRootNode());

    g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 4.0))));

    g_groundNode.reset(new SgRbtNode(RigTForm(Cvec3(0, g_groundY, 0))));
    g_groundNode->addChild(
        shared_ptr<MyShapeNode>(new MyShapeNode(g_ground, g_bumpFloorMat)));

    g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
    g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));

    constructRobot(g_robot1Node, g_redDiffuseMat);  // a Red robot
    constructRobot(g_robot2Node, g_blueDiffuseMat); // a Blue robot

    g_light1.reset(new SgRbtNode(RigTForm(Cvec3(4.0, 3.0, 5.0))));
    g_light2.reset(new SgRbtNode(RigTForm(Cvec3(-4, 1.0, -4.0))));
    g_light1->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_sphere, g_lightMat, Cvec3(0), Cvec3(0), Cvec3(0.5))));

    g_light2->addChild(shared_ptr<MyShapeNode>(
        new MyShapeNode(g_sphere, g_lightMat, Cvec3(0), Cvec3(0), Cvec3(0.5))));

    g_world->addChild(g_skyNode);
    g_world->addChild(g_groundNode);
    g_world->addChild(g_robot1Node);
    g_world->addChild(g_robot2Node);
    g_world->addChild(g_light1);
    g_world->addChild(g_light2);

    g_currentCameraNode = g_skyNode;
    
    // create a single transform node for both the bunny and the bunny
    // shells
    g_bunnyNode.reset(new SgRbtNode());
    
    g_bunnyNode.reset(new SgRbtNode(RigTForm(Cvec3(1,0,0))));

    // add bunny as a shape nodes
    g_bunnyNode->addChild(
        shared_ptr<MyShapeNode>(new MyShapeNode(g_bunnyGeometry, g_bunnyMat)));

    // add each shell as shape node
    for (int i = 0; i < g_numShells; ++i) {
        g_bunnyNode->addChild(shared_ptr<MyShapeNode>(
            new MyShapeNode(g_bunnyShellGeometries[i], g_bunnyShellMats[i])));
    }
    // from this point, calling g_bunnyShellGeometries[i]->upload(...) will
    // change the geometry of the ith layer of shell that gets drawn


    g_world->addChild(g_bunnyNode);
}

static void initAnimation() {
    g_animator.attachSceneGraph(g_world);
    g_curKeyFrame = g_animator.keyFramesBegin();
}

// Specifying shell geometries based on g_tipPos, g_furHeight, and g_numShells.
// You need to call this function whenver the shell needs to be updated

static Cvec2 getTexture(int vtx){
    Cvec2 helper[] = {Cvec2(0,0), Cvec2(g_hairyness, 0.0), Cvec2(0.0, g_hairyness)};
    return helper[vtx];
}

//static Cvec3 multiplier(RigTForm Oinv, Cvec3f tip_w){
//    RigTForm tipWorldRbt = RigTForm(Cvec3((double)tip_w[0], (double)tip_w[1], (double)tip_w[2]));
//    RigTForm tipObjRbt = Oinv * tipWorldRbt;
//
//}

static Cvec3f DtoF(Cvec3 a){
    return Cvec3f(float(a[0]), float(a[1]), float(a[2]));
}

static Cvec3 FtoD(Cvec3f a){
    return Cvec3(double(a[0]), double(a[1]), double(a[2]));
}

static void updateShellGeometry() {
    
    RigTForm Oinv = inv(getPathAccumRbt(g_world, g_bunnyNode));
    
    for (int shell = 0; shell < g_numShells; shell++){
        VertexPNX shellGeometry[g_bunnyMesh.getNumFaces() * 3];
        for (int face = 0; face < g_bunnyMesh.getNumFaces(); face++){
            for (int vtx = 0; vtx < 3; vtx++){
                Mesh::Vertex bvtx = g_bunnyMesh.getFace(face).getVertex(vtx);
                
                Cvec3 t = (Oinv * RigTForm(g_tipPos[3 * face + vtx])).getTranslation();
                Cvec3 n = (t - bvtx.getNormal())/ g_numShells;
                Cvec3 p = bvtx.getPosition() + (n * shell);
                Cvec2 tex = getTexture(vtx);
                
                shellGeometry[3 * face + vtx] = VertexPNX(p,n,tex);
            }
        }
        g_bunnyShellGeometries[shell]->upload(&shellGeometry[0], g_bunnyMesh.getNumFaces() * 3);
    
    }
    // TASK 1 and 3 TODO: finish this function as part of Task 1 and Task 3
}

// calculate s with p as with regular P, then convert p = ovject cordinate and s = to world coordinate.

// LUIS: I attempted to follow task 2, not working somewhere.

// New function to update the simulation every frame
static void hairsSimulationUpdate() {
    // TASK 2 TODO: write dynamics simulation code here
    
    RigTForm O = getPathAccumRbt(g_world, g_bunnyNode);
    for (int face = 0; face < g_bunnyMesh.getNumFaces(); face++){
        for (int vtx = 0; vtx < 3; vtx++){
            int idx = face * 3 + vtx;

            Mesh::Vertex v = g_bunnyMesh.getFace(face).getVertex(vtx);
            
            Cvec3 p = (O * RigTForm(v.getPosition())).getTranslation();
            Cvec3 s = p + (O * RigTForm(v.getNormal())).getTranslation() * g_furHeight;
            
            Cvec3 vel = g_tipVelocity[idx];
            Cvec3 t = g_tipPos[idx];

            Cvec3 force = g_gravity + (s - t) * g_stiffness;
            t = t + (vel * g_timeStep);
            t = p + (normalize(t-p) * g_furHeight);
            
            vel = (vel + (force * g_timeStep)) * g_damping;
            
            g_tipPos[idx] = t;
            g_tipVelocity[idx] = vel;
        }
    }
}

static void initSimulation() {
    g_tipPos.resize(g_bunnyMesh.getNumFaces() * 3, Cvec3(0));
    g_tipVelocity = g_tipPos;

    // TASK 1 TODO: initialize g_tipPos to "at-rest" hair tips in world
    // coordinates
    
    RigTForm O = getPathAccumRbt(g_world, g_bunnyNode);
    
    for (int face = 0; face < g_bunnyMesh.getNumFaces(); face++){
        for (int vtx = 0; vtx < 3; vtx++){
            Mesh::Vertex v = g_bunnyMesh.getFace(face).getVertex(vtx);
            Cvec3 s = v.getPosition() + (v.getNormal() * g_furHeight);
            g_tipPos[face * 3 + vtx] = (O * RigTForm(s)).getTranslation();
        }
    }
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
        for (int i = 0; i< g_simulationsPerSecond; i++){
            hairsSimulationUpdate();
        }
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
        initSimulation();
        initAnimation();

        glfwLoop();
        return 0;
    } catch (const runtime_error &e) {
        cout << "Exception caught: " << e.what() << endl;
        return -1;
    }
}
