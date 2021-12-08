
function new_img = draw_box (img, ci, cj, box_size, r, g, b)

    new_img = img;

    [ m n d ] = size(img);

    i_min = max(1, round(ci) - box_size);
    i_max = min(m, round(ci) + box_size);

    j_min = max(1, round(cj) - box_size);
    j_max = min(n, round(cj) + box_size);

    for i =  i_min:i_max
        for j =  j_min:j_max
            new_img(i, j, 1) = r;
            new_img(i, j, 2) = g;
            new_img(i, j, 3) = b;
        end
    end
end

