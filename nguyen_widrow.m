%inisialisasi nguyen widrow
unit_input=3;
unit_hidden=3;
beta=0.7*(unit_hidden)^(1/unit_input);

%set bobot v
vij=[rand(unit_input,unit_hidden)-0.5];

%hitung vij
vij_abs = sqrt(sum(sum(vij.^2)));
%update bobot
for i = 1:unit_input
    for j = 1:unit_hidden
        vij(i,j) = beta*vij(i,j)*(1/vij_abs);
    end
end

%set bobot bias
v0j=[rand(1,unit_hidden)-beta];

%set bobot w
wij=[rand(unit_input,unit_hidden)-0.5];

%hitung wij
wij_abs = sqrt(sum(sum(wij.^2)));
%update bobot
for i = 1:unit_input
    for j = 1:unit_hidden
        wij(i,j) = beta*wij(i,j)*(1/wij_abs);
    end
end

