import random
import numpy as np
import matplotlib.pyplot as plt

#class knot

def random_knot(grid=10, distribution='uniform'):
    x_rows = np.random.permutation(grid_size)
    o_columns = np.random.permutation(np.arange(1,grid_size))
    o_columns = np.append(o_columns,0)
    return (x_rows, o_columns)

def crossing_number(knot):
    x_rows = knot[0]
    o_columns = knot[1]
    grid_size = len(x_rows)
    passing_matrix = np.zeros((grid_size,grid_size))
    crossing_counter = 0
    for i in range(grid_size):
        #x_coordinates = (x_rows[i], o_columns[i-1])
        #o_coordinates_vert = (x_rows[i-1], o_columns[i-1])
        #o_coordinates_horiz = (x_rows[i], o_columns[i])
        min_vert = min(x_rows[i]+1,x_rows[i-1])
        max_vert = max(x_rows[i],x_rows[i-1]+1)
        min_horiz = min(o_columns[i-1],o_columns[i]+1)
        max_horiz = max(o_columns[i-1]+1,o_columns[i])
        for j in range(min_vert,max_vert):
            if passing_matrix[j,o_columns[i-1]] != 0:
                crossing_counter += 1
            passing_matrix[j,o_columns[i-1]] += 1
        for j in range(min_horiz,max_horiz):
            if passing_matrix[x_rows[i],j] != 0:
                crossing_counter += 1
            passing_matrix[x_rows[i],j] += 1
        #print(passing_matrix)
    return crossing_counter

def knot_matrix(knot):
    x_rows = knot[0]
    o_columns = knot[1]
    grid_size = len(x_rows)
    matrix = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        x_cell = (x_rows[i], o_columns[i-1])
        o_cell = (x_rows[i-1], o_columns[i-1])
        matrix[x_cell] = 1
        matrix[o_cell] = -1
    return matrix

def generate_batch(grid_size=10, batch_size=100, representation_knot='matrix', representation_cross='int', max_cross=35):
    knot = random_knot(grid_size)
    crossings = crossing_number(knot)
    if representation_knot == 'matrix':
        knot = knot_matrix(knot).flatten()
    if representation_knot == 'path_choice':
        knot = np.append(knot[0],knot[1])
    batch_knots = np.array([knot])
    if representation_cross == 'vec':
        cross = [int(i==crossings) for i in range(max_cross)]
    elif representation_cross =='int':
        cross = crossings
    batch_crossings = np.array([cross])

    for _ in range(batch_size-1):
        knot = random_knot(grid_size)
        crossings = crossing_number(knot)

        if representation_cross == 'vec':
            cross = [int(i==crossings) for i in range(max_cross)]
        elif representation_cross =='int':
            cross = crossings

        if representation_knot == 'matrix':
            knot = knot_matrix(knot).flatten()
        elif representation_knot == 'path_choice':
            knot = np.append(knot[0],knot[1])
        batch_knots = np.append(batch_knots,[knot],0)
        batch_crossings = np.append(batch_crossings, [cross], 0)
    return {'knot':batch_knots, 'crossings':batch_crossings}

