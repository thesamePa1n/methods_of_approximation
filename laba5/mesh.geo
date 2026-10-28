ls = 0.03;
a = 1;
b = 1;
r = 0.25;

Point(1) = {0, 0, 0, ls};
Point(2) = {a, 0, 0, ls};
Point(3) = {a, b, 0, ls};
Point(4) = {0, b, 0, ls};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Point(5) = {a/2, b/2, 0, ls};
Point(6) = {a/2 - r, b/2, 0, ls};
Point(7) = {a/2, b/2 + r, 0, ls};
Point(8) = {a/2 + r, b/2, 0, ls};
Point(9) = {a/2, b/2 - r, 0, ls};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Curve Loop(1) = {3, 4, 1, 2};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};
Plane Surface(2) = {2};

Physical Curve(1) = {4};
Physical Curve(2) = {3};
Physical Curve(3) = {2};
Physical Curve(4) = {1};

Physical Surface(1) = {1};
Physical Surface(2) = {2};