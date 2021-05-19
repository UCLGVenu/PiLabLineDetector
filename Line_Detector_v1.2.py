import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

img = cv2.imread('Big1_top_36.tif', 0)

line_test_val = 50 

line_test_arr = []

for i in range(len(img[line_test_val])):
    line_test_arr.append(img[line_test_val][i])

line_test_max = max(line_test_arr)
line_test_min = min(line_test_arr)
denoised = cv2.fastNlMeansDenoising(img, None, 50, 7, 21)
line_thresh_min = line_test_min + 0.25 * (line_test_max - line_test_min)

dummy, thresh1 = cv2.threshold(denoised, line_thresh_min, line_test_max, cv2.THRESH_BINARY)

cv2.imshow('Frame', thresh1)
cv2.waitKey(0)

edges = cv2.Canny(thresh1, 200, 250) #edge detection between 150 and 230, results in a black background and "top ground"
edges1 = cv2.Canny(img, 200, 250)

height = img.shape[0] - 150 #cuts the image
width = img.shape[1]

cv2.imshow('Frame', edges)
cv2.waitKey(0)

division = (height // 500)
test_arr = []
for i in range(1, 501):
    test_arr.append(division * i)

result = dict()


for val in test_arr:   #finds the position of the edges in terms of pixel
    arr = []
    false = []
    for j in range(width):
        if edges[val][j] > 200:
            arr.append(j)
    for i in range(1, len(arr)):
        if arr[i] - arr[i-1] < 5:
            false.append(arr[i-1])
    if false:
        for val2 in false:
            arr.remove(val2)
    result[val] = arr




#keys = list(result.keys())
#num_points = len(result[keys[0]]) #how many points to count (1st left edge, 1st right edge, 2nd left edge, etc)

#for v in result.values():
#    if len(v) != num_points:
#        print('Oh no!')
#        print(len(v))
#       break
keys = list(result.keys())
num_points = len(result[keys[0]])
np_list = [len(x) for x in result.values()]
np_counter = Counter(np_list)
num_points = np_counter.most_common(1)[0][0]
#print(num_points)
for key in keys:
    if len(result[key]) != num_points:
        result.pop(key)
keys = list(result.keys())
print(len(keys))



if thresh1[keys[100]][list(result.values())[100][0] + 2] > 200:
    white = True
else: white = False


total_sum = [0] * num_points
sum_tophalf = [0] * num_points
sum_bothalf = [0] * num_points
sum_distance = [0] * (num_points - 1) #of course, the distances are less than points

for i in range(len(keys)):
    for j in range(num_points):
        total_sum[j] += result[keys[i]][j]
        if i < len(keys)//2:
            sum_tophalf[j] += result[keys[i]][j]
        if i >= len(keys)//2:
            sum_bothalf[j] += result[keys[i]][j]
        if j > 0:
            v1 = result[keys[i]][j]
            v2 = result[keys[i]][j-1]
            sum_distance[j-1] += (v1 - v2)

avg_midpoints = [x / len(keys) for x in total_sum]
avg_tophalf = [x / (len(keys)//2) for x in sum_tophalf]
avg_bothalf = [x / (len(keys)//2) for x in sum_bothalf]
avg_distance = [x / len(keys) for x in sum_distance]
dist2 = [avg_midpoints[i] - avg_midpoints[i-1] for i in range(1, len(avg_midpoints))]

top_half_pixel = sum(keys[:len(keys)//2])/(len(keys)//2)
bot_half_pixel = sum(keys[len(keys)//2:])/(len(keys)//2)
pixel_gap = bot_half_pixel - top_half_pixel

top_bot_diff = [avg_tophalf[i] - avg_bothalf[i] for i in range(len(avg_tophalf))]
avg_diff = sum(top_bot_diff)/len(top_bot_diff)

theta = np.arctan(avg_diff/pixel_gap)

# Theta is CLOCKWISE angle FROM vertical.

true_pix_distance = [np.cos(theta) * x for x in avg_distance]

d1 = [true_pix_distance[x] for x in range(len(true_pix_distance)) if x%2 == 0]
d2 = [true_pix_distance[x] for x in range(len(true_pix_distance)) if x%2 == 1]

avg_d1 = sum(d1)/len(d1)
avg_d2 = sum(d2)/len(d2)
"""
if avg_d1 > avg_d2:
    print('Avg Large Spacing in pixels = ', avg_d1)
    print('Avg Small Spacing in pixels = ', avg_d2)
elif avg_d2 > avg_d1:
    print('Avg Large Spacing in pixels = ', avg_d2)
    print('Avg Small Spacing in pixels = ', avg_d1)
else:
    print('Wow, you got lucky!')
    print('Both spacings = ', avg_d1)
"""
# H = [680:700] for the scale number

img2 = edges1[700:-30, 15:200]
#img2_scaled = cv2.resize(img2, None, fx = 5, fy = 5, interpolation=cv2.INTER_LINEAR)
#cv2.imshow('Frame', img2)
#cv2.waitKey(0)

h2 = img2.shape[0]
w2 = img2.shape[1]

arr = []
for i in range(h2):
    for j in range(w2):
        if img2[i][j] > 0:
            arr.append(j)

arr_counter = Counter(arr)
scale_val = arr_counter.most_common(2)
scale_left = scale_val[0][0]
scale_right = scale_val[1][0]
scale_pixels = abs(scale_right - scale_left) - 2

set_scale = 200e-9
pixel_length = set_scale / scale_pixels

avg_d1 = avg_d1 * pixel_length
avg_d2 = avg_d2 * pixel_length
d1_nm = np.multiply(d1,pixel_length)
d2_nm = np.multiply(d2,pixel_length)


"""
if avg_d1 > avg_d2:
    print('Avg Large Spacing in m = ', round(avg_d1, 9), '+- ', round(np.std(d1_nm), 9))
    print('Avg Small Spacing in m = ', round(avg_d2, 9), '+- ', round(np.std(d2_nm), 9))
elif avg_d2 > avg_d1:
    print('Avg Large Spacing in m = ', round(avg_d2, 9), '+- ', round(np.std(d2_nm), 9))
    print('Avg Small Spacing in m = ', round(avg_d1, 9), '+- ', round(np.std(d1_nm), 9))
else:
    print('Wow, you got lucky!')
    print('Both spacings = ', round(avg_d1,9), '+- ', round(np.std(d1_nm), 9))
"""

dist = dict()
for k in result.keys():
    a = result[k]
    b=[]
    for i in range(len(a)-1):
        b.append(a[i+1]-a[i])
    b = np.multiply(b, pixel_length)
    b = np.multiply(b, np.cos(theta))
    dist[k] = b

hist1 = []
hist2 = []

for k in dist.keys():
    for i in range(len(dist[k])):
        if i%2 == 0 : hist1.append(dist[k][i])
        else: hist2.append(dist[k][i])
        
        
    
'''
n, bins, patches = plt.hist(x=hist1, bins='auto')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Width')
plt.ylabel('Counts')
plt.title('d1')
plt.show()
n, bins, patches = plt.hist(x=hist2, bins='auto')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Width')
plt.ylabel('Counts')
plt.title('d2')
plt.show()
'''

# Roughness Calculators

# Height = Pixel height - 150
# avg_midpoints = avg point of line

result2 = result.copy()

for k in result2.keys():
    for i in range(len(result2[k])):
        result2[k][i] -= (avg_midpoints[i] - (k - (height/2))*np.tan(theta))

line_dict = dict()

for i in range(len(result2[keys[0]])):
    line_dict[i] = []
    for j in range(len(result2)):
        line_dict[i].append(result2[keys[j]][i])

#keys2 = [x for x in line_dict.keys()]

#print(len(line_dict))

#corrs = []
"""
for k, v in line_dict.items():
    fft1 = np.fft.fft(v)
    fft2 = np.conj(np.fft.fft(v))
    correlation = np.fft.ifft(np.multiply(fft1, fft2))
    plt.bar(np.arange(0, len(v), 1), correlation)
    plt.title('Line '+str(int(k)+1))
    plt.show()
"""
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

'''for k, v in line_dict.items():
    correlation = autocorr(v)
    plt.bar(np.arange(0, len(correlation), 1), correlation)
    plt.title('Line '+str(int(k)+1))
    plt.show()'''
    
merged_lines = []

for v in line_dict.values():
    for values in v:
        merged_lines.append(values)
        
roughnessFFT = np.fft.fft(merged_lines)
rough2 = np.fft.fftshift(roughnessFFT)

plt.plot(roughnessFFT[0:len(roughnessFFT)//2])
plt.show()        

plt.plot(rough2[len(rough2)//2:])
plt.title('Shifted')
plt.show()
"""
print(list(roughnessFFT).index(max(roughnessFFT[1:len(roughnessFFT
                                                    )//2])) * pixel_length)
"""
np.savetxt("FTroughness_test.csv", np.real(roughnessFFT), delimiter = '\t')

summa = 0
means = []
for i in range(len(line_dict[1])):
    for j in range(len(line_dict)):
        summa += line_dict[j][i]
    means.append(summa*pixel_length/len(line_dict[j]))
    summa = 0

plt.plot(means)
plt.show()

displacement = sum(means)/len(means)
print(round(displacement, 10))

mid_white = []
mid_black = []
if (white == 1):
    for i in range(len(avg_midpoints)-1):
        if (i%2 == 0): mid_white.append((avg_midpoints[i+1] + avg_midpoints[i])/2)
        else: mid_black.append((avg_midpoints[i+1] + avg_midpoints[i])/2)
else:
     for i in range(len(avg_midpoints)-1):
        if (i%2 == 0): mid_black.append((avg_midpoints[i+1] + avg_midpoints[i])/2)
        else: mid_white.append((avg_midpoints[i+1] + avg_midpoints[i])/2)
        
grating_pitch = []
for i in range(len(mid_white)-1):
    grating_pitch.append(mid_white[i+1] - mid_white[i])
    
grating_pitch = np.multiply(grating_pitch, (np.cos(theta)*pixel_length))

if white: grating_width = d1_nm
else: grating_width = d2_nm

print ("The grating pitch is ", round(sum(grating_pitch)/len(grating_pitch), 9), "m +- ", round(np.std(grating_pitch), 9), "m")
print ("The grating width is ", round(sum(grating_width)/len(grating_width), 9), "m +- ", round(np.std(grating_width), 9), "m")

