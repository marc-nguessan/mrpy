#for(( i = 0; i <= 3; i++ ));do gnuplot -e "tt=$i" adap_grid_ign.gpi ; done
#mencoder "mf://*.png" -mf fps=5 -o ignition.avi -ovc lavc -lavcopts vcodec=wmv2
#mencoder "mf://*.png" -mf fps=5 -o ignition.avi -ovc lavc -lavcopts vcodec=mjpeg

#file = sprintf("finest_grid.dat")
#outfile = sprintf("finest_grid.png")
#file = sprintf("adapted_grid.dat")
#outfile = sprintf("adapted_grid.png")
file = sprintf("u_computed_adapted_grid_t_00551.dat")
outfile = sprintf("u_computed_adapted_grid_t_00551.png")
set term png enhanced size 700, 700
set output outfile
#set multiplot layout 1, 2 #title time(i) font "bold,14"
#set multiplot layout 1, 2 #title time(i)

set pm3d map
#set palette model XYZ rgbformulae 7,5,10
#set palette model HSV
#set palette rgb 3,2,1
#set palette cubehelix start 6 cycles -1 saturation 0.9
#set cbrange #[-1.:1.2]

set palette defined ( 0 'dark-blue',\
                        2.5 'light-blue',\
                        3.0 'light-green',\
                        4.3 'yellow',\
                        6 'red',\
                        7 'dark-red')
unset key
set size square 
#set object 1 rect from graph 0, 0 to graph 1, 1 behind fc rgb "black" fillstyle solid 1.0
set xtics out
set ytics out
#set cbrange [300:2200]
#set zrange  [0:2200] 
set xlabel "x"
set ylabel "y"
#set zlabel "T"
#set xtics -1,0.5,1
#set ytics -1,0.5,1
#set title "Temperature [K]" font "bold,14"
set title "Champ 2D"
#set label  "phi" at graph 0.75 ,graph 0.75  front
plot file using 1:($2):($6) notitle w p palette ps 0.7 pt 5
#splot for [ level = 1 : 10 ] file(tt) u 1:(-$2):( $3 == level ? ($5*(300-1000)+1000) :1/0  ) w p ps (8.0*10./2**(level)) pt 5 lc palette
#set title "Adaptive Grid" font "bold,14"
#set title "Adaptive Grid"
#set cbrange [3:10]
#plot for [ level = 1 : 10 ] file(tt) u 1:( $3 == level ? (-$2) :1/0  ):3 w p ps (8.0*10./2**(level)) pt 5 lc palette
show output
#unset multiplot
