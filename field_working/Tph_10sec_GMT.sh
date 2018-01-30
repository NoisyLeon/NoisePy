gmtset MAP_FRAME_TYPE fancy 
surface ./field_working/Tph_10sec -T0 -G./field_working/Tph_10sec.grd -I0.2 -R85.0/133.0/23.0/52.0 
grd2xyz ./field_working/Tph_10sec.grd -R85.0/133.0/23.0/52.0 > ./field_working/Tph_10sec.HD 
