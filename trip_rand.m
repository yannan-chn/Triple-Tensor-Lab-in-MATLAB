function tenv = trip_rand(sz,randFun)
% trip_rand  generates random factor tensors ([A,B,C] with i.i.d. elements)
% of a triple tensor. 
%
% Input:
%     sz        ---   [I,J,K,L], tensor size (I,J,K), triple rank L
%     randFun   ---   generator of randon variables
% Output:
%     tenv      ---   [a_IMN,b_LJN,C_LMK] factors are collected in a vector
%
% Yannan Chen  ... June 1, 2024

tenv = randFun(sum(sz(1:3))*sz(4)*sz(4));



