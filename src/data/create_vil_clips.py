import os
import glob
import tqdm
import time
import tables
import shutil
import argparse
import numpy as np
from socket import gethostname

def write(reader, in_array, out_array, num_in=13, num_out=12):
    # now actually write the data
    step = num_in
    n, nr, nc, nz = reader.root.IN.shape
    print(f'number of arrays : {n}')
    IN  = reader.root.IN.read()
    print(f'IN : {IN.shape}')
    for y in tqdm.tqdm(IN):
        start = stop = mid = 0
        for ii in np.arange(3):
            mid = start + step
            stop = start + 2*step - 1
            A = np.expand_dims(y[...,start:mid],axis=0)
            B = np.expand_dims(y[...,mid:stop],axis=0)
            in_array.append(A)
            out_array.append(B)
            start = start + step - 1
    return

def write_rows(reader, in_array, out_array, num_in=13, num_out=12):
    # now actually write the data
    step = num_in
    n, nr, nc, nz = reader.root.IN.shape
    print(f'number of arrays : {n}')
    IN  = reader.root.IN.read()
    print(f'IN : {IN.shape}')
    for y in tqdm.tqdm(IN):
        start = stop = mid = 0
        for ii in np.arange(3):
            mid = start + step
            stop = start + 2*step - 1
            A = np.expand_dims(np.reshape(y[...,start:mid], (nr*nc*num_in)), axis=0)
            B = np.expand_dims(np.reshape(y[...,mid:stop], (nr*nc*(num_out))), axis=0)
            in_array.append(A)
            out_array.append(B)
            start = start + step - 1
    return

def main(args):
    _, output_name = os.path.split(args.output)
    if not os.path.isdir(args.local_dir):
        os.makedirs(args.local_dir)
    temp_file = os.path.join(args.local_dir, output_name)
    print(f'temporary file on {gethostname()} : {temp_file}')
    
    with tables.open_file(temp_file, mode='w') as writer:
        # create arrays
        if not args.row:
            in_array = writer.create_earray("/", "IN",
                                                atom=tables.UInt8Atom(),
                                                shape=(0, 384, 384, args.num_in),
                                                chunkshape=(1,384,384,1),
                                                expectedrows=48000)
            out_array = writer.create_earray("/", "OUT",
                                                atom=tables.UInt8Atom(),
                                                shape=(0, 384, 384, args.num_out),
                                                chunkshape=(1,384,384,1),
                                                expectedrows=48000)        
            with tables.open_file(args.input, mode='r') as reader:
                write(reader, in_array, out_array, num_in=args.num_in, num_out=args.num_out)
        else:
            in_array = writer.create_earray("/", "IN",
                                                atom=tables.UInt8Atom(),
                                                shape=(0, 384*384*args.num_in))
            out_array = writer.create_earray("/", "OUT",
                                                atom=tables.UInt8Atom(),
                                                shape=(0, 384*384*args.num_out))
            with tables.open_file(args.input, mode='r') as reader:
                write_rows(reader, in_array, out_array, num_in=args.num_in, num_out=args.num_out)
            
    print('moving file to output dir')
    if os.path.isfile(args.output):
        print(f'output file {args.output} exists')
        backup_file = args.output+str(np.random.randint(1000, 100000, 1)[0])
        print(f'existing file renamed to {backup_file}')
        shutil.move(args.output, backup_file)
    shutil.move(temp_file, args.output)
    print(f'moved to : {args.output}')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input file', default='')
    parser.add_argument('--output', type=str, help='output file', default='')
    parser.add_argument('--num_in', type=int, default=13, help='number of input frames')
    parser.add_argument('--num_out', type=int, default=12, help='number of output frames')
    parser.add_argument('--local_dir', type=str, default='/state/partition1/user/ssamsi')
    parser.add_argument('--compression', type=str, default=None)
    parser.add_argument('--row', action='store_true', help='whether or not to write data in rows')
    args = parser.parse_args()

    main(args)
    
    print('all done')
    
