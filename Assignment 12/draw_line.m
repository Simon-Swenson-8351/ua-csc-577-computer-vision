% 
% This function draws the line represented by the second argument which is
% vector [a,b,d] that describes a line by a*i+b*j+d = 0. It returns a copy of the
% image (first argument) with the line drawn into the. The last 4 arguments are
% the width of the line, and its color in R, G, and B. 
%
% Previously, this name was given to a different function. That function is now
% called "draw_segment".
%
function new_image = draw_line(image, coef, width, red, green, blue)
   % Clip the line against the box, and then draw it.

   new_image = image; 

   m = size(image,1);
   n = size(image,2);

   num_found = 0;
   p = [];

   a = coef(1); b = coef(2); d = coef(3);

   if (abs(b) > 1e-5) 
       j_test = - (d + a) / b;

       % if ((j_test >= 1) & (j_test <= n))
       if ((j_test > -width) & (j_test < n+width))
          num_found = num_found + 1;
          p(num_found, : ) = [ 1  j_test ];
       end 

       j_test = - (d + a*m) / b;

       % if ((j_test >= 1) & (j_test <= n)) 
       if ((j_test > -width) & (j_test < n+width)) 
          num_found = num_found + 1;
          p(num_found, : ) = [ m  j_test ];
       end  
   end  

   if (abs(a) > 1e-5) 
       if (num_found < 2) 
           i_test = - (d + b) / a;

           if ((i_test > -width) & (i_test < m+width))
           % if ((i_test >= 1) & (i_test <= m))
              num_found = num_found + 1;
              p(num_found, : ) = [ i_test  1 ];
           end 
       end 
           
       if (num_found < 2) 
           i_test = - (d + n*b) / a;

           if ((i_test > -width) & (i_test < m+width)) 
           % if ((i_test >= 1) & (i_test <= m)) 
              num_found = num_found + 1;
              p(num_found, : ) = [ i_test  n ];
           end 
       end 
   end 

   if (num_found == 2) 
       new_image = draw_segment(new_image, p(1,:), p(2,:), width, red, green, blue);
   end 
end
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

