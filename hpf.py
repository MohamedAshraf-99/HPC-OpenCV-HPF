from mpi4py import MPI
from scipy import misc
import numpy as np
import cv2
# communicator object containing all processes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    # img = misc.face()
    # load image by giving its path
    img = cv2.imread(r'C:\Users\moham\OneDrive\Desktop\lena.jpeg', 0)

    # show image before filtering
    cv2.imshow('image', img)

    # split the image into sub_images
    sub_images = np.array_split(img, size-1, 0)

    for i in range(size-1):
        # receive the first sub_image then append the rest
        comm.send(sub_images[i], dest=i+1)

    for i in range(size-1):
        if i == 0:
            filtered = comm.recv(source=1)
        else:
            sub_image = comm.recv(source=i+1)
            # concatenate image sub_images
            filtered = np.concatenate((filtered, sub_image), axis=0)

    # show image after applying high pass filter
    cv2.imshow('filtered image', filtered)
    cv2.waitKey(0)

else:
    # each process receive image sub_image
    sub_image = comm.recv(source=0)

    # apply Laplacian or Sobel filter
    # Laplacian filter
    grad = cv2.Laplacian(sub_image, -1, ksize=1)
    hpf = cv2.convertScaleAbs(grad)

    # Sobel filter
    # hpf = cv2.Sobel(sub_image, -1, 1, 0, ksize=3)

    # send the filtered sub_image back to the master
    comm.send(hpf, dest=0)


# run by using the command
# mpiexec -n 4 python hpf.py