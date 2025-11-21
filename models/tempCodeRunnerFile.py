
        1: '1',2: '2', 3: '3',4: '4',
        5: '5',6: '6',7: '7',8: '8',
        9: '9',10: '9a',11: '10',12: '10a',
        13: '11',14: '12',15: '12a',16: '13',
        17: '14',18: '14a',19: '15',20: '16',
        21: '16a',22: '16b',23: '16c',
        24: '12b',25: '14b'
    }
    model = nnUnet(args).to("cuda")
    ls = torch.ones((10,1,args["image_shape"][0],args["image_shape"][1])).float().to("cuda")
    outs = model(ls)
    for out in outs:
        print(out.shap