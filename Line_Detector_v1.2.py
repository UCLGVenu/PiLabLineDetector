import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Import Image

img = cv2.imread('D:\\Documents\\UCL\\Project 1\\ZZ-Sandbox Testing\\Alessandro\\Big1_top_36.tif', 0)

# Set Threshold values for image

line_test_val = 50  # Choose height to test for threshold

line_test_arr = []

for i in range(len(img[line_test_val])):
    line_test_arr.append(img[line_test_val][i])

line_test_max = max(line_test_arr)  # Extract Max/Min values of tested line - use to Threshold
line_test_min = min(line_test_arr)

denoised = cv2.fastNlMeansDenoising(img, None, 50, 7, 21)  # Fast Denoise

# Set threshold at Min Value + 25% of difference

line_thresh_min = line_test_min + 0.25 * (line_test_max - line_test_min)

dummy, thresh1 = cv2.threshold(denoised, line_thresh_min, line_test_max, cv2.THRESH_BINARY)

# Show image (if required)

# cv2.imshow('Frame', thresh1)
# cv2.waitKey(0)

edges = cv2.Canny(thresh1, 200,
                  250)  # edge detection between 150 and 230, results in a black background and "top ground"
edges1 = cv2.Canny(img, 200, 250)

height = img.shape[0] - 150  # Cuts bottom part of image to exclude scale bar
width = img.shape[1]

# Show detected lines

cv2.imshow('Frame', edges)
cv2.waitKey(0)

# Draw horizontal lines through 'edges' - find positions of white pixels in each line

num_test_lines = 500    # Should be less that ~650.
division = (height // num_test_lines)
test_points = []  # Heights at which horizontal lines cut image
for i in range(1, num_test_lines + 1):
    test_points.append(division * i)

result = dict()
# result = Dictionary with Keys = Heights Tested, Values = Position of white pixels/edges.

for val in test_points:  # finds the position of the edges in terms of pixel
    arr = []
    false = []
    for j in range(width):  # Append white points to
        if edges[val][j] > 200:
            arr.append(j)
    for i in range(1, len(arr)):  # This + next loop removes double pixels/'fat' edges - takes furthest right point
        if arr[i] - arr[i - 1] < 5:
            false.append(arr[i - 1])
    if false:
        for val2 in false:
            arr.remove(val2)
    result[val] = arr

'''
result_keys = list(result.result_keys())
num_points = len(result[result_keys[0]]) #how many points to count (1st left edge, 1st right edge, 2nd left edge, etc)

for v in result.values():
   if len(v) != num_points:
    print('Oh no!')
    print(len(v))
    break
'''

# Section finds any anomalous lines by finding most common number of edges detected in each tested height,
# and discarding any lines that do not have this number of edges.

result_keys = list(result.keys())  # Keys of Results - i.e. the heights tested
np_list = [len(x) for x in result.values()]
np_counter = Counter(np_list)
num_points = np_counter.most_common(1)[0][0]

for key in result_keys:
    if len(result[key]) != num_points:
        result.pop(key)
result_keys = list(result.keys())
print(len(result_keys))

# Test if first line is white or not.

if thresh1[result_keys[100]][list(result.values())[100][0] + 2] > 200:
    white = True
else:
    white = False

# Find average position of lines, and position of 'top half' and 'bottom half' of lines. Used to calculate angle.

total_sum = [0] * num_points
sum_tophalf = [0] * num_points
sum_bothalf = [0] * num_points
sum_distance = [0] * (num_points - 1)  # of course, the distances are less than points

for i in range(len(result_keys)):
    for j in range(num_points):
        total_sum[j] += result[result_keys[i]][j]
        if i < len(result_keys) // 2:
            sum_tophalf[j] += result[result_keys[i]][j]
        if i >= len(result_keys) // 2:
            sum_bothalf[j] += result[result_keys[i]][j]
        if j > 0:
            v1 = result[result_keys[i]][j]
            v2 = result[result_keys[i]][j - 1]
            sum_distance[j - 1] += (v1 - v2)

avg_midpoints = [x / len(result_keys) for x in total_sum]
avg_tophalf = [x / (len(result_keys) // 2) for x in sum_tophalf]
avg_bothalf = [x / (len(result_keys) // 2) for x in sum_bothalf]
avg_distance = [x / len(result_keys) for x in sum_distance]

# Find average position of Top/Bottom Half pixel

top_half_pixel = sum(result_keys[:len(result_keys) // 2]) / (len(result_keys) // 2)
bot_half_pixel = sum(result_keys[len(result_keys) // 2:]) / (len(result_keys) // 2)
pixel_gap = bot_half_pixel - top_half_pixel

# Find average offset between the top half/bottom half pixel values

top_bot_diff = [avg_tophalf[i] - avg_bothalf[i] for i in range(len(avg_tophalf))]
avg_diff = sum(top_bot_diff) / len(top_bot_diff)

# Calculate Theta

theta = np.arctan(avg_diff / pixel_gap)

# Theta is CLOCKWISE angle FROM vertical.
# Correct distances for angle

true_pix_distance = [np.cos(theta) * x for x in avg_distance]

d1 = [true_pix_distance[x] for x in range(len(true_pix_distance)) if x % 2 == 0]
d2 = [true_pix_distance[x] for x in range(len(true_pix_distance)) if x % 2 == 1]

avg_d1 = sum(d1) / len(d1)
avg_d2 = sum(d2) / len(d2)
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
# ----------------------------------------------------------------------------------------------------------------------
# Section used to find pixel - meter scaling

# H = [680:700] for the scale number

img2 = edges1[700:-30, 15:200]  # Section of image with scale bar

'''
These lines show scaled area of scale bar - mostly for debugging.
# img2_scaled = cv2.resize(img2, None, fx = 5, fy = 5, interpolation=cv2.INTER_LINEAR)
# cv2.imshow('Frame', img2)
# cv2.waitKey(0)
'''

h2 = img2.shape[0]
w2 = img2.shape[1]

arr = []
for i in range(h2):
    for j in range(w2):
        if img2[i][j] > 0:
            arr.append(j)

# Scale bar detected by taking most common positions of edges (in y) in scale bar region - gives ends of scale bar.

arr_counter = Counter(arr)
scale_val = arr_counter.most_common(2)
scale_left = scale_val[0][0]
scale_right = scale_val[1][0]
scale_pixels = abs(scale_right - scale_left) - 2    # -2 Factor due to quirk of code to ensure accurate conversion

set_scale = 200e-9
pixel_length = set_scale / scale_pixels

avg_d1 = avg_d1 * pixel_length
avg_d2 = avg_d2 * pixel_length
d1_nm = np.multiply(d1, pixel_length)
d2_nm = np.multiply(d2, pixel_length)

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
    b = []
    for i in range(len(a) - 1):
        b.append(a[i + 1] - a[i])
    b = np.multiply(b, pixel_length)
    b = np.multiply(b, np.cos(theta))
    dist[k] = b

hist1 = []
hist2 = []

for k in dist.keys():
    for i in range(len(dist[k])):
        if i % 2 == 0:
            hist1.append(dist[k][i])
        else:
            hist2.append(dist[k][i])

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

# ---------------------------------------------------------------------------------------------------------------------
# Edge Analysis
# Height = Pixel height - 150
# avg_midpoints = avg point of line

result2 = result.copy()

# Normalise deviations from calculated 'mean-edge' position - i.e. normalise deviations to be around 0.

for k in result2.keys():
    for i in range(len(result2[k])):
        result2[k][i] -= (avg_midpoints[i] - (k - (height / 2)) * np.tan(theta))

line_dict = dict()
# line_dict has keys 0 to (number of lines - 1) and values = deviations from mean lines normalised to 0

for i in range(len(result2[result_keys[0]])):
    line_dict[i] = []
    for j in range(len(result2)):
        line_dict[i].append(result2[result_keys[j]][i])

# Code for auto-correlation for each line

"""
for k, v in line_dict.items():
    fft1 = np.fft.fft(v)
    fft2 = np.conj(np.fft.fft(v))
    correlation = np.fft.ifft(np.multiply(fft1, fft2))
    plt.bar(np.arange(0, len(v), 1), correlation)
    plt.title('Line '+str(int(k)+1))
    plt.show()
"""

'''
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]



for k, v in line_dict.items():
    correlation = autocorr(v)
    plt.bar(np.arange(0, len(correlation), 1), correlation)
    plt.title('Line '+str(int(k)+1))
    plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------
# Merge lines to carry out Fourier Transform on merged set to test for randomness of variation

merged_lines = []

for v in line_dict.values():
    for values in v:
        merged_lines.append(values)

roughnessFFT = np.fft.fft(merged_lines)
rough2 = np.fft.fftshift(roughnessFFT)

plt.plot(roughnessFFT[0:len(roughnessFFT) // 2])
plt.show()

plt.plot(rough2[len(rough2) // 2:])
plt.title('Shifted')
plt.show()

'''
print(list(roughnessFFT).index(max(roughnessFFT[1:len(roughnessFFT
                                                    )//2])) * pixel_length)

np.savetxt("FTroughness_test.csv", np.real(roughnessFFT), delimiter='\t')
'''

# Test if mean deviation = 0 (as expected)

summa = 0
means = []
for i in range(len(line_dict[1])):
    for j in range(len(line_dict)):
        summa += line_dict[j][i]
    means.append(summa * pixel_length / len(line_dict[j]))
    summa = 0

plt.plot(means)
plt.show()

displacement = sum(means) / len(means)
print(round(displacement, 10))

# Get grating pitches (full width of white+black section) + get grating width (width of white section only)

mid_white = []
mid_black = []
if (white == 1):
    for i in range(len(avg_midpoints) - 1):
        if (i % 2 == 0):
            mid_white.append((avg_midpoints[i + 1] + avg_midpoints[i]) / 2)
        else:
            mid_black.append((avg_midpoints[i + 1] + avg_midpoints[i]) / 2)
else:
    for i in range(len(avg_midpoints) - 1):
        if (i % 2 == 0):
            mid_black.append((avg_midpoints[i + 1] + avg_midpoints[i]) / 2)
        else:
            mid_white.append((avg_midpoints[i + 1] + avg_midpoints[i]) / 2)

grating_pitch = []
for i in range(len(mid_white) - 1):
    grating_pitch.append(mid_white[i + 1] - mid_white[i])

grating_pitch = np.multiply(grating_pitch, (np.cos(theta) * pixel_length))

if white:
    grating_width = d1_nm
else:
    grating_width = d2_nm

print("The grating pitch is ", round(sum(grating_pitch) / len(grating_pitch), 9), "m +- ",
      round(np.std(grating_pitch), 9), "m")
print("The grating width is ", round(sum(grating_width) / len(grating_width), 9), "m +- ",
      round(np.std(grating_width), 9), "m")
