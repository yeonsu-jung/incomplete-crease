function x0 = find_critical_points(xx,yy)

pp = spline(xx,yy);
dfdx = fnder(pp,1);
x0 = fnzeros(dfdx);

end