import os
from PIL import Image
import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt as distrans
import tcripser

def extract_vessel_features(featurename='betti_PHT',
                            isSeleton=True,
                            root='/scr2/jiacyao/TopoAly/Mouse/',
                            vesselpath='skeletons',
                            savepath ='topofeatures'):
    vessel = sorted([f for f in os.listdir(vesselpath)])
    saveroot = os.path.join(root, savepath)
    if not os.path.isdir(saveroot):
        os.makedirs(saveroot)
    extraction = {
        'betti_PHT': Betti_PHT(vessel), # Betti curve for 7-direction filtrations
        'betti': Betti_ending(vessel), # Betti curve
        'PI': PI_ending(vessel), # Persistence Image
        'PI_local': PI_Local(vessel) # Patchwise Persistence Image
    }
    try:
        extraction[featurename]
    except KeyError:
        print('Feature name not valid')
    return
    

def PI_Local(masks):
    gen_PI = vector_methods.PersistenceImage(bandwidth=0.1, weight=lambda x: x[1]/0.6 if x[1]<0.6 else 1,
                                             resolution=[20, 20], im_range=[0,1,0,1])
    for f in masks:
        pds = []
        name = f.split('.')[0]
        print(name)

        # Modify the following code to extract patchwise persistence image for 2D image
        '''
        '''
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(maskpath, f))).astype(np.float32)
        # Break 3D image into tiles
        k,h,w = image.shape
        K, M, N = k//8, h//8, w//8
        k = k - k%8
        h = h - h%8
        w = w - w%8
        tiles = [image[z:z+K, x:x+M, y:y+N] for z in range(0,k,K) for x in range(0,h,M) for y in range(0,w,N)]
        '''
        '''

        for tile in tiles:
            if np.sum(tile):
                skeleton = skeletonize(tile, method='lee')
                origins = find_one_degrees(skeleton)
                if len(origins[0]):
                    distance_map = bfs_distance(tile, origins, metric='geodesic')
                else:
                    distance_map = tile/255
            else:
                distance_map = tile/255
            distance_map = np.where(distance_map<0, 1, distance_map)
            pd = get_PD(distance_map, name)
            pds.append(pd)
        pis = gen_PI.transform(pds)
        #print(pis.shape)
        #pis = np.reshape(pis, (1, 8, 160, 160))
        np.save(os.path.join(saveroot, 'pi_' + name + '.npy'), pis)

def Betti_ending(masks):
    for f in masks:
        name = f.split('.')[0]
        print(name)
        # Modify the following code to extract betti curve for 2D image
        '''
        '''
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(maskpath, f))).astype(np.float32)
        '''
        '''

        skeleton = skeletonize(image, method='lee')
        origins = find_one_degrees(skeleton)
        if len(origins[0]):
            distance_map = bfs_distance(image, origins, metric='geodesic')
        else:
            distance_map = image
        distance_map = np.where(distance_map<0, 1, distance_map)
        betti = get_betticurve(distance_map, name)
        np.save(os.path.join(saveroot, 'betti_' + name + '.npy'), betti)

def Betti_PHT(masks):
    for f in masks:
        bettis = []
        name = f.split('.')[0]
        print(name)
        # Modify the following code to extract betti curve for 2D image
        '''
        '''
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(maskpath, f))).astype(np.float32)
        '''
        '''

        directions = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
        for v in directions:
                distance_map = scan(image, v)
                #plot_map(distance_map, name)
                betti = get_betticurve(distance_map, name)
                bettis.append(betti)
        bettis = np.asarray(bettis).flatten()
        np.save(os.path.join(saveroot, 'betti_' + name + '.npy'), bettis)
    print(bettis.shape)

def scan(image, v):
    '''
    generate distance map for a given direction
    '''
    k,h,w = image.shape
    distance_map = np.zeros((k,h,w))
    for i in range(k):
        for j in range(h):
            for l in range(w):
                # calculate inner product of v and (i+1,j+1,l+1)
                dis = v[0]*(i+1) + v[1]*(j+1) + v[2]*(l+1)
                distance_map[i,j,l] = dis
    distance_map = np.where(image==0, -1, distance_map)
    max_dis = np.max(distance_map)
    distance_map = np.where(distance_map<0, 1, distance_map/max_dis)
    return distance_map

def PI_ending(masks):
    gen_PI = vector_methods.PersistenceImage(bandwidth=0.1, weight=lambda x: x[1]/0.6 if x[1]<0.6 else 1,
                                             resolution=[20, 20], im_range=[0,1,0,1])
    for f in masks:
        pds = []
        name = f.split('.')[0]
        print(name)

        # Modify the following code to extract persistence image for 2D image
        '''
        '''
        image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(maskpath, f))).astype(np.float32)
        '''
        '''
        skeleton = skeletonize(image, method='lee')
        origins = find_one_degrees(skeleton)
        if len(origins[0]):
            distance_map = bfs_distance(image, origins, metric='geodesic')
        else:
            distance_map = image
        distance_map = np.where(distance_map<0, 1, distance_map)
        pd = get_PD(distance_map, name)
        pds.append(pd)
        pis = gen_PI.transform(pds)
        #print(pis.shape)
        np.save(os.path.join(saveroot, 'pi_' + name + '.npy'), pis)

def find_one_degrees(image):
    '''
    function to find one-degree points in a skeleton
    '''
    z,h,w = image.shape
    kernel = np.ones((3,3))
    positions = [[], [], []]
    for i in range(h):
        for j in range(w):
            for k in range(z):
                if image[k,i,j] == 255 and _degree(image, k, i, j) == 1:
                    positions[0].append(k)
                    positions[1].append(i)
                    positions[2].append(j)
    return positions

def _degree(image, k, i, j):
    '''
    function to calculate the degree of a point in a skeleton
    '''
    k,h,w = image.shape
    neighbor = [(kk,ii,jj) for kk in range(k-1, k+2) for ii in range(i-1,i+2) for jj in range(j-1,j+2)]
    sum = 0
    for (kk, ii, jj) in neighbor:
        try:
            if image[kk, ii, jj] == 255:
                sum += 1
        except:
            pass 
    return sum == 1

def bfs_distance(skeleton, origins, metric='geodesic'):
    '''
    function to calculate distance map from a set of origins
    '''
    D,H,W = skeleton.shape
    map = -np.ones((D, H, W))
    visited = set()
    ox, oy, oz = origins[0][0], origins[1][0], origins[2][0]
    for i in range(len(origins[0])):
        x, y, z = origins[0][i], origins[1][i], origins[2][i]
        map[x][y][z] = 0
        visited.add((x,y,z))
    while len(origins[0]):
        i, j, k = origins[0].pop(0), origins[1].pop(0), origins[2].pop(0)
        for (ni, nj, nk) in [(ii,jj,kk) for kk in range(k-1, k+2) for ii in range(i-1,i+2) for jj in range(j-1,j+2)]:
            if ((ni, nj, nk) not in visited) and 0<=ni<D and 0<=nj<H and 0<=nk<W and skeleton[ni, nj, nk] == 1:
                visited.add((ni, nj, nk))
                origins[0].append(ni)
                origins[1].append(nj)
                origins[2].append(nk)
                if metric == 'geodesic':
                    map[ni][nj][nk] = map[i][j][k] + 1
    max_dis = np.max(map)
    if max_dis == 0:
        return map
    map = map/(max_dis)
    return map

def get_PD(map, name):
    '''
    function to calculate persistence diagram from a distance map
    '''
    #print('Computing persistence homology...')
    start = time.time()
    pd = tcripser.computePH(map, maxdim=1)
    end = time.time()
    pd = pd[pd[:,2]<=1,1:3]
    return pd

def get_betticurve(map, name):
    '''
    function to calculate betti curve from a distance map
    '''
    #print('Computing persistence homology...')
    start = time.time()
    pd = tcripser.computePH(map, maxdim=1)
    end = time.time()

    pds = [pd[pd[:,0] == i] for i in range(2)]
    # fig, ax = plt.subplots()
    # persim.plot_diagrams([p[:,1:3] for p in pds], ax=ax)
    # plt.savefig('./figures/diagram_' + name + '.png')
    filtration = np.arange(0, 1, 0.01)
    betti_curve0 = []
    betti_curve1 = []
    for f in filtration:
        betti0 = [1 for p in pd if (p[0]==0 and p[1] <= f and p[2] >= f)]
        betti1 = [1 for p in pd if (p[0]==1 and p[1] <= f and p[2] >= f)]
        betti_curve0.append(np.sum(betti0))
        betti_curve1.append(np.sum(betti1))
    betti_curve0, betti_curve1 = np.asarray(betti_curve0), np.asarray(betti_curve1)
    #print('timing: {:.2f}'.format(end-start))
    betti = np.hstack((betti_curve0, betti_curve1))
    return betti

if __name__ == '__main__':
    featurenames = ['betti_PHT', 'betti', 'PI', 'PI_local']
    root = 'path/to/root'
    vesselpath = 'masks'
    savepath = 'topofeatures'
    extract_vessel_features(featurename=featurenames[0],
                            isSeleton=True,
                            root=root,
                            vesselpath=vesselpath,
                            savepath=savepath)