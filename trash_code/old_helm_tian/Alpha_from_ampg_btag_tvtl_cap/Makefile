INST_DIR = $(HOME)/bin
BIN  = alpha_from_ampg_btag_tvtl

fflags = -O2 -Wall
cflags = -O2 -Wall

LDLIBS =

FFLAGS = $(DBG) $(fflags)

CFLAGS = $(DBG) $(cflags)

FC = gfortran

CC = g++

DBG = -g

all : $(BIN)

#FOBJS = driver_c.o whiten_phamp.o fft_phamp.o swapn.o
FOBJS =  driver.o grdt_smth.o grdt_beta.o lplc.o Gauss_Smoothing.o calc_dist_azim.o fit_line.o least_squares_sine.o discard.o

$(BIN) : $(FOBJS)
	$(CC) $(CFLAGS) $(FOBJS) -o $(BIN) $(LDLIBS) 

install : $(BIN)
	install -s $(BIN) $(INST_DIR)

clean :
	rm -f $(BIN) core $(FOBJS)

