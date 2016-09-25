import numpy as np
import cv2

# Stolen from http://stackoverflow.com/questions/3252194/numpy-and-line-intersections
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1

def angle(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    return abs(np.arctan2(da[0], da[1]) - np.arctan2(db[0], db[1]))

def dist(a1, b1):
    d = a1 - b1
    return np.sqrt(d[0] ** 2 + d[1] ** 2)

def findCorners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blur, 100, 250)
    cv2.imshow('frame', edges)
    minLineLength = 200
    maxLineGap = 10000000
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, None, minLineLength, maxLineGap)
    backtorgb = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    if lines != None:
        corners = []
        for i in range(0, len(lines[0])):
            for j in range(i + 1, len(lines[0])):
                a1 = np.array([lines[0][i][0], lines[0][i][1]])
                a2 = np.array([lines[0][i][2], lines[0][i][3]])
                b1 = np.array([lines[0][j][0], lines[0][j][1]])
                b2 = np.array([lines[0][j][2], lines[0][j][3]])
                if angle(a1, a2, b1, b2) > np.pi / 8:
                    corner = map(int, seg_intersect(a1, a2, b1, b2))
                    if abs(corner[0] - 500) < 500 and abs(corner[1] - 500) < 500:
                        newCorner = True
                        for otherCorner in corners:
                            if dist(np.array(corner), np.array(otherCorner)) < 10:
                                newCorner = False
                        if newCorner:
                            corners.append(corner)
        if len(corners) == 4:
            for corner in corners:
                cv2.circle(backtorgb, tuple(corner), 10, [0,255,0], -1)
            cv2.imshow('frame', backtorgb)
            return corners
    return None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    corners = findCorners(frame)
    if corners:
        for corner in corners:
            cv2.circle(frame, tuple(corner), 10, [0,255,0], -1)
    print(corners)
    # cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
