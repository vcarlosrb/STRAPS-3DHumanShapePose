import numpy as np
import trimesh
import math

def getMinX(polygon):
    return polygon.bounds[0]

def getMinY(polygon):
    return polygon.bounds[1]

def getMaxX(polygon):
    return polygon.bounds[2]

def getMaxY(polygon):
    return polygon.bounds[3]

def getArmpits(sections):
    location_percentage = 76
    approximate_location = math.floor(location_percentage*len(sections)/100)
    section_min = approximate_location - 10
    section_max = approximate_location + 10
    range_sections = range(section_min, section_max)
    armpits = None
    position = None
    length = None
    stop = False
    for index in range_sections:
        if stop == False:
            if len(sections[index].entities) == 1:
                armpits = sections[index]
                position = index
                length = sections[index].polygons_closed[0].length
                stop = True
                
    return armpits, position, length

def getChest (sections, armpits_location):
    cont = armpits_location
    stop = False
    minimum = 100
    chest = None
    position = None
    length = None
    while stop == False:
        polygon = getLargerAreaPolygon(sections[cont])
        minimumPolygonX = getMinX(polygon)
        if minimumPolygonX < minimum:
            minimum = minimumPolygonX
            chest = sections[cont]
            position = cont
            length = polygon.length
        else:
            stop = True
        cont = cont - 1
        
    return chest, position, length

def getCrotch(sections):
    location_percentage = 47 # porcentaje
    approximate_location = math.floor(location_percentage*len(sections)/100)
    section_min = approximate_location - 15
    section_max = approximate_location + 15
    range_sections = range(section_min, section_max)
    
    crotch = None
    position = None
    length = None
    stop = False
    for index in range_sections:
        if stop == False:
            if len(sections[index].entities) == 1:
                crotch = sections[index]
                position = index
                length = sections[index].polygons_closed[0].length
                stop = True
                
    return crotch, position, length

def getHip(sections, crotch_location):
    cont = crotch_location
    stop = False
    maximum = 0
    hip = None
    position = None
    length = None
    while stop == False:
        polygon = getLargerAreaPolygon(sections[cont])
        maximumPolygonX = getMaxX(polygon)
        if maximumPolygonX > maximum:
            maximum = maximumPolygonX
            hip = sections[cont]
            position = cont
            length = polygon.length
        else:
            stop = True
        cont = cont + 1
        
    return hip, position, length

def getSections(mesh, levels):
    sections = mesh.section_multiplane(plane_origin=mesh.centroid,
                                   plane_normal=[0,1,0], 
                                   heights=levels)
    new_sections = []
    cont = 0
    for item in sections:
        if item != None:
            new_sections.append(item)
            cont = cont + 1
    return new_sections
                                
def getLargerAreaPolygon(section):
    higher = 0
    polygon = None
    for pol in section.polygons_closed:
        if pol != None:
            if pol.area > higher:
                higher = pol.area
                polygon = pol
    return polygon

def getHeight(mesh):
    slice_r = mesh.section(plane_origin=mesh.centroid, 
                     plane_normal=[0,0,1])
    slice_2D, _ = slice_r.to_planar()
    minY = slice_2D.bounds[0][1]
    maxY = slice_2D.bounds[1][1]
    return (maxY - minY) # En metros

def getWeight(mesh):
    body_density = 0.985
    return (mesh.volume*1000) * body_density

def getHeightsWithBatchSize(vertices, faces):
    heights = []
    for index in range(vertices.shape[0]):
        mesh = trimesh.Trimesh(vertices[index].cpu().detach().numpy(), faces)
        heights.append(getHeight(mesh))
    return heights

def getBodyMeasurementWithBatchSize(vertices, faces, batch_size):
    weights = []
    heights = []
    if vertices.shape[0] == batch_size:
        for index in range(0, vertices.shape[0]-1):
            mesh = trimesh.Trimesh(vertices[index].cpu().detach().numpy(), faces)
            weights.append(getWeight(mesh))
            heights.append(getHeight(mesh))

    return weights, heights

def getBodyMeasurement(vertices, faces):
    steps = 0.005
    levels = np.arange(-1.5, 1.5, step=steps)
    vertices = vertices.cpu().detach().numpy()
    vertices = vertices.reshape(vertices.shape[1], vertices.shape[2])
    mesh = trimesh.Trimesh(vertices, faces)
    sections = getSections(mesh, levels)
    weight = getWeight(mesh)
    height = getHeight(mesh)
    _, armpits_loaction, _ = getArmpits(sections)
    _, chest_location, chest_length = getChest(sections, armpits_loaction)
    _, crotch_location, _ = getCrotch(sections)
    _, hip_loaction, hip_length = getHip(sections, crotch_location)

    return weight, height, chest_length, hip_length