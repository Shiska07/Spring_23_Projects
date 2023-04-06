clc;
clear;

p7 = [-4.96 4.48];
p10 = [-1.44 -2.88];

p7_7 = dot(p7,p7)
p10_10 = dot(p10,p10)
p7_10 = dot(p7,p10)

a7 = 0.03;
w = (p7.*a7) + (p10.*(-1*a7));

b = - (dot(w,p10)+dot(w,p7))./2;

% second iteration
p1 = [1.44 2.08];

p1_1 = dot(p1, p1)
p10_1 = dot(p1, p10)
p1_7 = dot(p1, p7)

