architecture = [
    {"k":3,"s":1,"p":1},
    {"k":3,"s":1,"p":1},
    {"k":2,"s":2,"p":0,"maxpool":True},

    {"k":3,"s":1,"p":1},
    {"k":3,"s":1,"p":1},
    {"k":2,"s":2,"p":0,"maxpool":True},

    {"k":3,"s":1,"p":1},
    {"k":3,"s":1,"p":1},
    {"k":2,"s":2,"p":0,"maxpool":True},

    {"k":3,"s":1,"p":1},
    {"k":3,"s":1,"p":1},
    {"k":2,"s":2,"p":0,"maxpool":True},

    {"k":3,"s":1,"p":1},
    {"k":3,"s":1,"p":1},
    {"k":2,"s":2,"p":0,"maxpool":True},

    {"k":3,"s":1,"p":1},
    {"k":3,"s":1,"p":1},
    {"k":2,"s":2,"p":0,"maxpool":True},

    {"k":3,"s":1,"p":1},
    {"k":3,"s":1,"p":1},
    {"k":2,"s":2,"p":0,"maxpool":True},
]
init_image_size = 512
j=1
r=1
start = 0.5
for i,l in enumerate(architecture):

    s = l["s"]
    k = l["k"]
    p = l["p"]

    new_j = j*s
    new_r = r + (k-1)*j
    new_start = start + ((k - 1) / 2 - p) * j
    r = new_r
    j = new_j
    start = new_start
    is_maxpool = l.get("maxpool")
    if(is_maxpool):
        init_image_size = init_image_size // 2
    print(f"Layer {i+1}: RF={new_r}, jump={new_j}, start={new_start} out size={init_image_size}")

