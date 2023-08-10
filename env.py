
def graphdict_setup(env_name,STATE_SPACE):
    graph = dict()
    office_edges = [(1, 56),
             (2, 56),
             (3, 56),
             (4, 56),
             (5, 58),
             (6, 58),
             (7, 58),
             (8, 45),
             (9, 45),
             (10, 45),
             (11, 46),
             (12, 47),
             (13, 47),
             (14, 47),
             (15, 48),
             (16, 59),
             (17, 60),
             (18, 60),
             (19, 59),
             (20, 59),
             (21, 47),
             (22, 23), (22, 45),
             (23, 22), (23, 44),
             (24, 45),
             (25, 44), (25, 26), (25, 50),
             (26, 25), (26, 53),
             (27, 54),
             (28, 54),
             (29, 54),
             (30, 54),
             (31, 55),
             (32, 55),
             (33, 55),
             (34, 56),
             (35, 56),
             (36, 56),
             (37, 54),
             (38, 54),
             (39, 54),
             (40, 54),
             (41, 58),
             (42, 58),
             (43, 44),
             (44, 43), (44, 58), (44, 25), (44,49),(44, 23),(44, 45), 
             (45, 8), (45, 9), (45, 10), (45, 46),(45, 22), (45, 24), (45, 44),
             (46, 11), (46, 47), (46, 45),
             (47, 46), (47, 12), (47, 13), (47, 14), (47, 48), (47, 21),
             (48, 15), (48, 47), (48, 59),
             (49, 44), (49, 60), (49, 50),
             (50, 49), (50, 25), (50, 51),(50, 53),
             (51, 52), (51, 50), 
             (52, 51),
             (53, 58), (53, 26), (53, 50), (53, 54),
             (54, 30), (54, 29), (54, 28), (54, 27), (54, 53), (54, 40), (54, 39), (54, 38), (54, 55),(54, 37), 
             (55, 33), (55, 32), (55, 31), (55, 54), (55, 56),
             (56, 34), (56, 55),(56, 35), (56, 36),(56, 57), (56, 1), (56, 2), (56, 3), (56, 4),
             (57, 56), (57, 58),
             (58, 57), (58, 5), (58, 6), (58, 7), (58, 41), (58, 42), (58, 44), (58, 53),
             (59, 48), (59, 16), (59, 60), (59, 19), (59, 20),
             (60, 18), (60, 17), (60, 49), (60, 59)]
    museum_edges = [
        (1, 2), (1, 5), (1, 9), (1, 7),
        (2, 1), (2, 3), (2, 4),
        (3, 2),
        (4, 2),
        (5, 1),
        (6, 7),
        (7, 6), (7, 8), (7, 1),
        (8, 7),
        (9, 1), (9, 10),
        (10, 9), (10, 11), (10, 12), (10, 13), (10, 18), (10, 19), (10, 20), (10, 27),
        (11, 10), (11, 14), (11, 28),
        (12, 10), (12, 15), (12, 16),
        (13, 10), (13, 17),
        (14, 11), (14, 15),
        (15, 14), (15, 12),
        (16, 12), (16, 17),
        (17, 16), (17, 13),
        (18, 10), (18, 19), (18, 26), (18, 70),
        (19, 10), (19, 18), (19, 20), (19, 25), (19, 26),
        (20, 10), (20, 19), (20, 21), (20, 24),
        (21, 20), (21, 22),
        (22, 21), (22, 23),
        (23, 22), (23, 24),
        (24, 23), (24, 20), (24, 25),
        (25, 24), (25, 19), (25, 26),
        (26, 25), (26, 19), (26, 18),
        (27, 10), (27, 49),
        (28, 11), (28, 29),
        (29, 28), (29, 34),
        (30, 31), (30, 33),
        (31, 30), (31, 32),
        (32, 33), (32, 31), (32, 37),
        (33, 32), (33, 36),
        (34, 29), (34, 35),
        (35, 34), (35, 36), (35, 40), (35, 49),
        (36, 35), (36, 33), (36, 39), (36, 37),
        (37, 36), (37, 32), (37, 38),
        (38, 37), (38, 42),
        (39, 36), (39, 40), (39, 41),
        (40, 35), (40, 39), (40, 43),
        (41, 39), (41, 42),
        (42, 41), (42, 38),
        (43, 40), (43, 44), (43, 47),
        (44, 43), (44, 45),
        (45, 44), (45, 46),
        (46, 45), (46, 47),
        (47, 43), (47, 46), (47, 48),
        (48, 47), (48, 50),
        (49, 35), (49, 27), (49, 58), (49, 50),
        (50, 48), (50, 49), (50, 51), (50, 52),
        (51, 50),
        (52, 50), (52, 56), (52, 53),
        (53, 52), (53, 54), (53, 55), (53, 56),
        (54, 53), (54, 55),
        (55, 54), (55, 53),
        (56, 52), (56, 53), (56, 64),
        (57, 58), (57, 63),
        (58, 49), (58, 57), (58, 59), (58, 62),
        (59, 60), (59, 61), (59, 58),
        (60, 70), (60, 61), (60, 59),
        (61, 60), (61, 59), (61, 69), (61, 68),
        (62, 58), (62, 67),
        (63, 57), (63, 66), (63, 64),
        (64, 63), (64, 56), (64, 65),
        (65, 64), (65, 66),
        (66, 65), (66, 67), (66, 63),
        (67, 62), (67, 66), (67, 68),
        (68, 61), (68, 67), (68, 69),
        (69, 61), (69, 68),
        (70, 18), (70, 60)]

    test_edges=[(1,0),(1,2),
                (2,1),(2,3),
                (3,2),(3,4),
                (4,3),(4,5),
                (5,4),(5,6),
                (6,5),(6,7),
                (7,6),(7,8),
                (8,7),(8,9),
                (9,8),(9,10),
                (10,9)]
    
    if env_name=="test_env":
        edges=test_edges
    elif env_name=='office':
        edges=office_edges
    elif env_name=='museum':
        edges=museum_edges
    else:raise NameError("Env is wrong!! Please choose right env_name")

    for i in range(STATE_SPACE):
        graph.update({i: [i]})
    for edge in edges:
        graph[edge[0]].append(edge[1])
    graph[0].append(1)
    return graph