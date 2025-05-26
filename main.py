import pygame
import numpy as np
import math
import random as rand

# Initialize Pygame, create a window, save pygame information
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
FPS = 60
dt = 0

# Graphic Settings
fov = 90 # Field of view
zfar = 1000 # Far plane
znear = 0.1 # Near plane

# Camera Settings
camPos = np.array([-40, 20, -22], dtype=float) # Camera position
lookDir = np.array([0, 0, 1], dtype=float) # Camera direction vector
up = np.array([0, 1, 0], dtype=float) # Up vector
fYaw = 0

def scaleValue(x, inMin, inMax, outMin, outMax):
    return (x - inMin) / (inMax - inMin) * (outMax - outMin) + outMin



class Object():
    def __init__(self, pos = np.array([0, 0, 0], dtype=float), rot = np.array([0, 0, 0], dtype=float), scaling = np.array([1, 1, 1], dtype=float)):
        self.pos = pos
        self.rot = rot
        self.scaling = scaling

    def setPos(self, pos):
        self.pos = np.array(pos, dtype=float)    

    def setRot(self, rot):
        self.rot = np.array(rot, dtype=float)

    def setScaling(self, scaling):
        self.scaling = np.array(scaling, dtype=float)

    def createFromObjFile(self, filename):
        with open(filename, "r") as file:
            content = file.readlines()
            tempVerts = np.empty((0, 3), dtype=float) 
            tempFaces = np.empty((0, 3), dtype=int) 

            for line in content:
                text = line.split(" ")
                if text[0] == "v":
                    # Füge die neuen Positionsdaten als Zeile hinzu
                    tempVerts = np.vstack((tempVerts, [float(text[1]), float(text[2]), float(text[3])]))
                elif text[0] == "f":
                    # Füge die neuen Face-Daten als Zeile hinzu
                    tempFaces = np.vstack((tempFaces, [int(text[1])-1, int(text[2])-1, int(text[3])-1]))
            
            self.vertices = tempVerts
            self.faces = tempFaces


    #4D vertices of the cube because we want to have homogeneous coordinates, to be able to use matmul for everything, including translation
    vertices = np.array([[]], dtype=float)

    faces = np.array([[]], dtype=float)

def translate(x, y, z, position):
    # Translate the vertex by the cube's position
    x += position[0]
    y += position[1]
    z += position[2]

    return x, y, z

def rotate(x, y, z, rotation):
    xAngle = rotation[0] # Rotation around the X axis
    yAngle = rotation[1] # Rotation around the Y axis
    zAngle = rotation[2] # Rotation around the Z axis

    # Rotation matrix around the Y axis
    xRotMatrix = makeRotationX(xAngle)
    yRotMatrix = makeRotationY(yAngle)
    zRotMatrix = makeRotationZ(zAngle)

    # Matrix Multiplication to rotate the vertex
    pos = np.array([x, y, z], dtype=float)
    pos = np.matmul(xRotMatrix, pos) # Rotate around X axis
    pos = np.matmul(yRotMatrix, pos) # Rotate around Y axis
    pos = np.matmul(zRotMatrix, pos) # Rotate around Z axis
    return pos

def makeRotationX(angle):
    angle = math.radians(angle)
    return np.array([
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)],
    ])
def makeRotationY(angle):
    angle = math.radians(angle)
    return np.array([
        [math.cos(angle), 0, math.sin(angle)],
        [0, 1, 0],
        [-math.sin(angle), 0, math.cos(angle)],
    ])
def makeRotationZ(angle):
    angle = math.radians(angle)
    return np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ])

def scale(x, y, z, scaling):
    # Scale the vertex by the cube's scaling factor
    x *= scaling[0]
    y *= scaling[1]
    z *= scaling[2]

    return x, y, z

def findIntersectionWithScreenBounds(insidePoint, outsidePoint):
    # Normalized device coordinates (NDC) bounds are -1 to 1 in all axes
    # We need to find where the line between insidePoint and outsidePoint intersects the NDC cube
    
    # We'll use a parametric approach to find the intersection
    direction = outsidePoint - insidePoint
    t_values = []
    
    # Check intersections with each plane of the NDC cube
    for i in range(3):  # x, y, z axes
        if direction[i] != 0:
            # Front plane (1) and back plane (-1)
            t_front = (1 - insidePoint[i]) / direction[i]
            t_back = (-1 - insidePoint[i]) / direction[i]
            
            # We only want intersections between the points (t between 0 and 1)
            if 0 <= t_front <= 1:
                t_values.append(t_front)
            if 0 <= t_back <= 1:
                t_values.append(t_back)
    
    # The smallest t value is the first intersection
    if not t_values:
        return False, None  # No intersection found (shouldn't happen in our case)
    
    t = min(t_values)
    intersection_point = insidePoint + t * direction
    
    return True, intersection_point

def applyClipping(face):
    # Method either returns normal face, 1 new face, 2 new faces or wants to skip
    # first return Value equals scenario: (0: normal face) (1: new face) (2: 2 new Faces) (-1: skip)
    status = 0
    newFaces = []

    insidePoints = []
    outsidePoints = []

    # check which points are in space
    for v in face:
        if -1 <= v[0] <= 1 and -1 <= v[1] <= 1 and -1 <= v[2] <= 1:
            insidePoints.append(v)
        else:
            outsidePoints.append(v)

    # All files are inside
    if len(insidePoints) == 3:
        newFaces = face
        status = 0
    # No points are inside -> return none
    elif len(insidePoints) == 0:
        status = -1
    # One point inside -> one new triangle
    elif len(insidePoints) == 1:
        inside = insidePoints[0]
        _, newPoint1 = findIntersectionWithScreenBounds(inside, outsidePoints[0])
        _, newPoint2 = findIntersectionWithScreenBounds(inside, outsidePoints[1])
        
        newFace = np.array([inside, newPoint1, newPoint2])
        newFaces = newFace
        status = 1
    # Two points inside -> two new triangles
    elif len(insidePoints) == 2:
        inside1, inside2 = insidePoints[0], insidePoints[1]
        outside = outsidePoints[0]
        
        _, newPoint = findIntersectionWithScreenBounds(inside1, outside)
        
        # Create two new triangles
        newFace1 = np.array([inside1, inside2, newPoint])
        newFace2 = np.array([inside2, newPoint, findIntersectionWithScreenBounds(inside2, outside)[1]])
        
        newFaces = [newFace1, newFace2]
        status = 2

    return status, newFaces



def lookAt(pos, target, up):
    # Calculate the forward vector
    newForward = target - pos
    newForward /= np.linalg.norm(newForward) # Normalize the forward vector

    #Calculate the new up vector
    a = newForward * np.dot(up, newForward) # Project the up vector onto the forward vector
    newUp = up - a # Subtract the projection from the up vector
    newUp /= np.linalg.norm(newUp) # Normalize the up vector

    # Calculate the right vector
    newRight = np.cross(up, newForward)

    # Construct Dimensioning and Translation Matrix
    translationMatrix = np.array([
        [newRight[0], newUp[0], newForward[0], 0.0],
        [newRight[1], newUp[1], newForward[1], 0.0],
        [newRight[2], newUp[2], newForward[2], 0.0],
        [-np.dot(pos, newRight), -np.dot(pos, newUp), -np.dot(pos, newForward), 1]
    ])

    return translationMatrix

def relativePosToWorld(vertex, position, rotation, scaling):
    # Scale the vertex by the cube's scaling factor
    x, y, z = scale(vertex[0], vertex[1], vertex[2], scaling)

    # Get Rotated Vertex Position
    x, y, z = rotate(x, y, z, rotation) 

    # Get Moved Vertex Position
    x, y, z = translate(x, y, z, position)

    return np.array([x, y, z, 1]) # Returns the vertex in world coordinates

def projectTo2D(vertex):
    a = screen.get_height() / screen.get_width() # aspect ration
    radFov = math.radians(fov) # math.tan works in radians, so we need to convert degrees to radians
    f = 1 / (math.tan(radFov / 2)) # focal length
    lam = zfar / (zfar - znear) # perspective projection matrix

    # Projection matrix
    projectionMatrix = np.array([
        [a * f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, lam, -lam * znear],
        [0, 0, 1, 0]
    ])

    # Multiply the vertex by the projection matrix
    projectedVertex = np.matmul(projectionMatrix, vertex)
    x = projectedVertex[0]
    y = projectedVertex[1]
    z = projectedVertex[2]

    # Projection Division
    w = projectedVertex[3]
    if (w != 0):
        # Perform perspective division
        x /= w
        y /= w
        z /= w
    
    return np.array([x, y, z]) # Returns NDC (Normalized Device Coordinates) of the vertex

def getScreenVertexPosition(pos):
        # Project the 3D vertex to 2D screen coordinates
        horMidpoint = screen.get_width() / 2
        verMidpoint = screen.get_height() / 2

        x = horMidpoint + (pos[0] * horMidpoint)
        y = verMidpoint - (pos[1] * verMidpoint) # Pygame renders from up to bottom, so we need to reverse the y axis
        return np.array([x, y])

def drawFace(object, fill=True, color=(0, 0, 0)):
    global lookDir

    # Loop through every face
    faces = object.faces
    vertices = object.vertices

    # First we save the faces in an array with the distance of their middleparts to the camera, so we can draw the furthest faces first
    facesToRoster = np.empty((0, 4), dtype=float) # Create an empty array to store the faces
    for face in faces:
        v1 = relativePosToWorld(vertices[face[0]], object.pos, object.rot, object.scaling)
        v2 = relativePosToWorld(vertices[face[1]], object.pos, object.rot, object.scaling)
        v3 = relativePosToWorld(vertices[face[2]], object.pos, object.rot, object.scaling)
        facesToRoster = np.vstack((facesToRoster, [face[0], face[1], face[2], (v1[2] + v2[2] + v3[2]) / 3]))
        
    # Now we sort the faces by their distance to the camera
    facesToRoster = facesToRoster[facesToRoster[:, 3].argsort()] # Sort the faces by their distance to the camera
    facesToRoster = np.flip(facesToRoster, 0) # Flip the array to have the faces in the right order

    # We need to move the vertices based on camera direction and position
    target = np.array(([0, 0, 1]))
    camRot = makeRotationY(fYaw)
    lookDir = np.matmul(camRot, target)
    target = camPos + lookDir
    matView = lookAt(camPos, target, up) # Create the camera matrix

    for face in facesToRoster:
        # Store the vertex ref in a variable
        idx1, idx2, idx3 = int(face[0]), int(face[1]), int(face[2])
        
        # Convert the relative vertex positions to world coordinates
        v1 = relativePosToWorld(vertices[idx1], object.pos, object.rot, object.scaling)
        v2 = relativePosToWorld(vertices[idx2], object.pos, object.rot, object.scaling)
        v3 = relativePosToWorld(vertices[idx3], object.pos, object.rot, object.scaling)

        # Convert world coordinates based on the camera position
        v1 = np.matmul(v1, matView)
        v2 = np.matmul(v2, matView)
        v3 = np.matmul(v3, matView)

        # We have 3 Points, we take the normal vector of the face and check if the normal vector shows to the camera
        # Calculate the normal vector of the face
        line1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
        line2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]
        normal = np.cross(line1, line2) # Cross product to get the normal vector

        # Normalize the normal vector through dividing by its length
        normalLength = np.linalg.norm(normal)
        if normalLength != 0:
            normal = normal / normalLength

        # Skalar product to check if the normal vector shows to the camera
        camToFace = np.array([v1[0] - camPos[0], v1[1] - camPos[1], v1[2] - camPos[2]]) # Vector from camera to face

        if np.dot(normal, camToFace) < 0:

            #Get Projected Position
            v1 = projectTo2D(v1)
            v2 = projectTo2D(v2)
            v3 = projectTo2D(v3)

            # Apply clipping
            status, clippedFaces = applyClipping(np.array([v1, v2, v3]))
            
            # Handle different clipping scenarios
            if status == -1:  # Skip this face
                continue
            elif status == 0:  # Original face
                faces_to_draw = [clippedFaces]
            elif status == 1:  # One new face
                faces_to_draw = [clippedFaces]
            elif status == 2:  # Two new faces
                faces_to_draw = clippedFaces

            # Draw all faces after clipping
            for face_to_draw in faces_to_draw:
                # Get Pixel Position for each vertex
                v1_screen = getScreenVertexPosition(face_to_draw[0])
                v2_screen = getScreenVertexPosition(face_to_draw[1])
                v3_screen = getScreenVertexPosition(face_to_draw[2])
                
                # Illumination
                lightDir = np.array([0, 0, -1]) # Direction of the light source
                lightDirLength = np.linalg.norm(lightDir) # Length of the light direction vector
                lightDir = lightDir / lightDirLength # Normalize the light direction vector

                dp = np.dot(normal, lightDir) # Dot product to get the angle between the normal vector and the light direction vector
                
                # Change color based on similarity to the light direction vector
                a = int(dp * 255)
                if a >= 0:
                    a = int(scaleValue(a, 0, 250, 60, 250))
                    illuminatedColor = pygame.Color(a, a, a)
                else:
                    a = int(scaleValue(-a, 0, 250, 60, 250))
                    illuminatedColor = pygame.Color(a, a, a)

                # [Rest des Zeichencodes bleibt gleich]
                if fill==True:
                    pygame.draw.polygon(screen, color, [v1_screen, v2_screen, v3_screen], 0)
                    pygame.draw.polygon(screen, (0, 0, 0), [v1_screen, v2_screen, v3_screen], 1)
                else:
                    pygame.draw.polygon(screen, illuminatedColor, [v1_screen, v2_screen, v3_screen], 1)


def generateSmallStars(amount):
    # Erzeuge alle Zufallswerte auf einmal
    random_values = np.random.uniform(-1, 1, (amount, 2))
    stars = np.zeros((amount, 3))
    
    for i in range(amount):
        stars[i, :2] = getScreenVertexPosition(random_values[i])  # x, y
        stars[i, 2] = np.random.uniform(0.1, 0.5)  # Größe (falls zufällig)
    
    return stars



videoShip = Object() # Create a videoShip at the origin
videoShip.setPos([20, 35, 50]) # Set the cube position to (0, 0, 5)
videoShip.setRot([0, -90, 0]) # Set the cube rotation to (0, 0, 0)
videoShip.setScaling([5, 5, 5])
videoShip.createFromObjFile("objects/VideoShip.obj")

teapot = Object() # Create a teapot at the origin
teapot.setPos([0, 0, 5]) # Set the position to (0, 0, 5)
teapot.setRot([0, 0, 0]) # Set the rotation to (0, 0, 0)
teapot.createFromObjFile("objects/teapot.obj")

axis = Object() # Create a teapot at the origin
axis.setPos([0, 0, 5]) # Set the position to (0, 0, 15)
axis.setRot([0, 0, 0]) # Set the rotation to (0, 0, 0)
axis.setScaling([2, 2, 1])
axis.createFromObjFile("objects/axis.obj")

mountain = Object() # Create a teapot at the origin
mountain.setPos([0, 0, 0]) # Set the position to (0, 0, 15)
mountain.setRot([0, 0, 0]) # Set the rotation to (0, 0, 0)
mountain.createFromObjFile("objects/mountains.obj")

saturn = Object() # Create a teapot at the origin
saturn.setPos([-188, 100, 102]) # Set the position to (0, 0, 15)
saturn.setRot([0, 0, 0]) # Set the rotation to (0, 0, 0)
saturn.setScaling([25, 25, 25]) # Set the rotation to (0, 0, 0)
saturn.createFromObjFile("objects/planet_with_ring.obj")

energyNodes = Object()
energyNodes.setPos([100, 100, 100]) # Set the position to (0, 0, 15)
energyNodes.setRot([0, 0, 0]) # Set the rotation to (0, 0, 0)
energyNodes.setScaling([15, 15, 15]) # Set the rotation to (0, 0, 0)
energyNodes.createFromObjFile("objects/octahedron.obj")

energyNodes2 = Object()
energyNodes2.setPos([127, 95, 100]) # Set the position to (0, 0, 15)
energyNodes2.setRot([0, 0, 0]) # Set the rotation to (0, 0, 0)
energyNodes2.setScaling([5, 5, 5]) # Set the rotation to (0, 0, 0)
energyNodes2.createFromObjFile("objects/octahedron.obj")

energyNodes3 = Object()
energyNodes3.setPos([75, 110, 100]) # Set the position to (0, 0, 15)
energyNodes3.setRot([0, 0, 0]) # Set the rotation to (0, 0, 0)
energyNodes3.setScaling([6, 6, 6]) # Set the rotation to (0, 0, 0)
energyNodes3.createFromObjFile("objects/octahedron.obj")

stars = generateSmallStars(100)


# Program Loop
while running:
    dt = clock.tick(FPS) / 1000

    videoShip.pos[0] -= 15 * dt
    videoShip.pos[1] -= 3 * dt
    videoShip.rot[0] += 20 * dt
    videoShip.rot[1] += 20 * dt
    videoShip.rot[2] += 20 * dt

    energyNodes.rot[0] += 20 * dt
    energyNodes.rot[1] += 40 * dt
    energyNodes.rot[2] += 20 * dt

    energyNodes2.rot[0] -= 40 * dt
    energyNodes2.rot[1] -= 60 * dt
    energyNodes2.rot[2] += 60 * dt

    energyNodes3.rot[0] += 40 * dt
    energyNodes3.rot[1] -= 60 * dt
    energyNodes3.rot[2] -= 60 * dt

    #Check whether player closed window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("black")

    keys = pygame.key.get_pressed()
    # Move the camera with the arrow keys
    if keys[pygame.K_UP]:
        camPos[1] += 5 * dt
    if keys[pygame.K_DOWN]:
        camPos[1] -= 5 * dt
    if keys[pygame.K_LEFT]:
        camPos[0] -= 5 * dt
    if keys[pygame.K_RIGHT]:
        camPos[0] += 5 * dt

    forwardVelocity = np.multiply(lookDir, 3.0 * dt)
    if keys[pygame.K_w]:
        camPos += forwardVelocity 
    if keys[pygame.K_s]:
        camPos -= forwardVelocity
    if keys[pygame.K_a]:
        fYaw -= 2.0
    if keys[pygame.K_d]:
        fYaw += 2.0

    
    if keys[pygame.K_i]:
        videoShip.pos[2] += 2 
    if keys[pygame.K_k]:
        videoShip.pos[2] -= 2
    if keys[pygame.K_j]:
        videoShip.pos[0] += 2
    if keys[pygame.K_l]:
        videoShip.pos[0] -= 2
    if keys[pygame.K_u]:
        videoShip.pos[1] -= 2
    if keys[pygame.K_u]:
        videoShip.pos[1] += 2



    # Draw stars in the background
    
    for star in stars:
        star[:2] += 1 * dt 
        if star[0] > screen.get_width():
            star[0] = 0
            star[1] = screen.get_height() - star[1] 

        if star[1] > screen.get_height():
            star[0] = screen.get_width() - star[0] 
            star[1] = 0

        pygame.draw.circle(screen, "white", (star[0], star[1]), star[2] * 6)


    drawFace(axis, False) # Draw the object
    drawFace(energyNodes, True, (139, 0, 0))
    drawFace(energyNodes2, True, (88, 138, 69))
    drawFace(energyNodes3, True, (158, 45, 106))
    drawFace(saturn, False) # Draw the object
    drawFace(videoShip, False)

    
  
    # Update the display
    pygame.display.flip()

pygame.quit()