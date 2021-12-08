% This is an example illustrating the use of quadprog to solve a simple linear
% system of equations with a single inequality constraint.
%
% Suppose the following needs to be solved
%
%     Ux = y, with the constraint x >= 1.
% 
% where U = [1 0; 0 1; 1 1; 1 -1], x = [x1, x2]' (unknown) and y = [1, 2, 3, 4]'.
% The constraint x >= 1 is a short way of writing [x1, x2] >= [1 1].
% 
% Since a solution is not always guaranteed to exist, we will solve it in the
% least squares sense, i.e., minimize the least square error:
%     
%                      (Ux - y)'(Ux - y) = x'(U'U)x - 2y'Ux + y'y      (1)
%
% with the constraint  x >= 1. Note that second term comes from the two terms
% -y'Ux and -x'Uy which are simple numbers (as are the other terms), and hence
% the second can be replaced by its transpose to get the same thing as the
% first.
%
% This can be done with the help of quadprog function. It might help to do 'help
% quadprog' before reading on. 
%
% We now do some algebra to get our problem into quadprog format. 
%
% First note that in (1) y'y is a constant, and so it does not factor in the
% solution.  Minimizing a function plus a constant is the same as minimizing it
% ignoring the constant.  Further, we could divide (1) by 2, and the value of x
% that gives the minimun would still be the same.  So, we could use the
% following objective function instead:
%
%                      (1/2)x'(U'U)x - y'Ux
%
% Our constraint [x1, x2] >= [1 1] is equivalent to:
%   
%                     A * x' <= [-1 -1]'
%
% where (quadprog variable) A is the negative of the identity. 
% We set b=[-1 -1]'. H is between the x' and the x so it is U'U. f' multiples
% the x in the second term, so f'=(-y'U) and f=-U'y. 
%
% Thus we get the following code. 
%

U = [1 0; 0 1; 1 1; 1 -1];
y = [1, 2, 3, 4]';

H = U'*U;
f = -U'*y;
A = [-1 0; 0 -1];
b = [-1, -1]';

x = quadprog(H, f, A, b)

