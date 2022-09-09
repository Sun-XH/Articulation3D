from pytorch3d.io import load_obj, save_obj
import cv2
import pdb
import numpy as np
from matplotlib import pyplot as plt
import torch


def project2D(pcd,h=480,w=640,focal_length=517.97):
# def project2D(pcd,h=480,w=640,focal_length=368.635):
    #pcd is Nx3
    offset_x = w/2
    offset_y = h/2
    K = [[focal_length, 0, offset_x],
        [0, focal_length, offset_y],
        [0, 0, 1]]

    #pdb.set_trace()
    if torch.is_tensor(pcd) and pcd.is_cuda:
        K = torch.FloatTensor(K).cuda()
        proj = (K@(pcd.T)).T
        proj = proj[:,:2] / proj[:,2][:,None]
        return proj 

    #proj is Nx2
    proj = (np.array(K)@(np.asarray(pcd).T)).T
    proj = proj[:,:2] / proj[:,2][:,None]
    return proj

verts, faces, _ = load_obj(
    '/localhome/xsa55/Xiaohao/Articulation3D/articulation3d/output_test_modified/frame_0000/arti_pred.obj')
h = 480
w = 640
x = verts[:, 0]
y = verts[:, 1]
z = verts[:, 2]
cx = w/2
cy = h/2
fx = 517.97
fy = 517.97

verts_2d = project2D(verts)

px = -x/z*fx + cx
py = y/z*fy + cy
# pdb.set_trace()
mask = np.zeros((h,w))

# pdb.set_trace()
for i in range(len(verts)):
    mask[int(py[i]), int(px[i])] = 1
    # mask[int(verts_2d[i, 1]), int(verts_2d[i, 0])] = 1

cap = cv2.VideoCapture("/localhome/xsa55/Xiaohao/Articulation3D/articulation3d/tools/demo/teaser.mp4")
step = 1
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    h, w, n = frame.shape
    # pdb.set_trace()
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # h, w, _ = frame.shape
    # im = cv2.resize(frame, (int(w*(517.97/983)),
    #                 int(h*(517.97/983))))
    # im = transform_image(im)
    frames.append(frame)

# pdb.set_trace()
# im = cv2.addWeighted(frames[0], 0.6, mask, 0.4, 0)
# cv2.imsave("test_image.png", im)



plt.figure()
plt.imshow(frames[0])
plt.imshow(mask, alpha=0.7)
plt.show()

