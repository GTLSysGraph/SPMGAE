
from     easydict        import EasyDict
import   torch
from     utils          import *
from     build_easydict import *
from     evaluation_mini_batch     import * 
from     datasets_dgl.data_dgl     import *

import   logging
from     tqdm import tqdm
from     models import build_model
import torch.nn.functional as F
from utils import compute_edge_sim,show_occupied_memory


def pretrain_mini_batch(param, model, graph, optimizer, batch_size, lr, weight_decay ,max_epoch, device, use_scheduler, model_dir, model_name):
    logging.info("start training SPMGAE mini batch node classification..")

    model = model.to(device)
    optimizer = create_optimizer(optimizer, model, lr, weight_decay)    

    # dataloader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    dataloader = dgl.dataloading.NodeDataLoader(
            graph,torch.arange(0, graph.num_nodes()), sampler,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=1)

    logging.info(f"After creating dataloader: Memory: {show_occupied_memory():.2f} MB")
    if use_scheduler and max_epoch > 0:
        logging.info("Use scheduler")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    # train
    for epoch in range(max_epoch):
        epoch_iter = tqdm(dataloader)
        loss_list = []
        for input_nodes, output_nodes, _ in epoch_iter:
            model.train()

            ###########################################################
            subgraph = dgl.node_subgraph(graph, input_nodes).to(device)
            subgraph = subgraph.to(device)
            n_node = subgraph.num_nodes()

            edges_sim       = compute_edge_sim(subgraph.edges(), subgraph.ndata["feat"], sim_mode=param.sim_mode)
            ref_edges_index = torch.where(edges_sim >= param.keep_threshold)
            del_edges_index = torch.where(edges_sim <  param.dele_threshold)

            edges           = torch.stack((subgraph.edges()[0],subgraph.edges()[1]),dim=0)
            ref_edges       = edges[:, ref_edges_index[0]]
            del_edges       = edges[:, del_edges_index[0]]
            graph_refine    = dgl.graph((ref_edges[0],ref_edges[1]), num_nodes=n_node).to(device)
            graph_refine = graph_refine.remove_self_loop()
            graph_refine = graph_refine.add_self_loop()

            # print('num  edges : {}'.format(subgraph.num_edges()))
            # print('keep edges : {}'.format(len(ref_edges_index[0])))
            # print('del  edges : {}'.format(len(del_edges_index[0])))
            ############################################################

            loss, loss_dict = model(graph_refine, del_edges, subgraph.ndata["feat"], epoch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            epoch_iter.set_description(f"train_loss: {loss.item():.4f}, Memory: {show_occupied_memory():.2f} MB")
            loss_list.append(loss.item())
            

        if scheduler is not None:
            scheduler.step()

        torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        print(f"# Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}, Memory: {show_occupied_memory():.2f} MB")
    return model




def Train_SPMGAE_nodecls(margs):
    #########################
    device = f"cuda" if torch.cuda.is_available() else "cpu"
  
    dataset_name = margs.dataset
    if dataset_name.split('-', 1)[0] == 'Attack':
        # dataset_name = dataset_name.split('-')[1]
        DATASET = EasyDict()
        DATASET.ATTACK = {
            "data":dataset_name,
            "attack":margs.attack.split('-')[0],
            "ptb_rate":margs.attack.split('-')[1]
        }
        # now just attack use
        dataset  = load_attack_data(DATASET['ATTACK'])
        graph  = dataset.graph   
        train_idx = graph.ndata['train_mask'].nonzero().squeeze()
        val_idx = graph.ndata['val_mask'].nonzero().squeeze()
        test_idx = graph.ndata['test_mask'].nonzero().squeeze()
        split_idx = {'train': train_idx, 'valid': val_idx,'test': test_idx}
        labels = graph.ndata['label']
    else:
        dataset  = load_data(dataset_name)
        split_idx = dataset.get_idx_split()
        graph, labels = dataset[0] # ogbn-arxiv

    
    labels = labels.squeeze()   # 变成1维，后面验证需要
    graph = dgl.remove_self_loop(graph)


    # print(graph)
    # homophily_score_before = compute_graph_homophily_score(graph, graph.ndata['label'])
    # print(homophily_score_before)
    # quit()

    graph = dgl.add_self_loop(graph)


    num_classes = dataset.num_classes
    num_features = graph.ndata['feat'].shape[1]

    MDT = build_easydict()
    param         = MDT['MODEL']['PARAM']

    if param.use_cfg:
        param = load_best_configs(param, dataset_name.split('-', 1)[1].lower() if dataset_name.split('-', 1)[0] == 'Attack' else dataset_name.lower() , "./configs.yml")
    print(param)

    seeds         = param.seeds
    max_epoch     = param.max_epoch
    max_epoch_f   = param.max_epoch_f
    num_hidden    = param.num_hidden
    num_layers    = param.num_layers
    encoder_type  = param.encoder
    decoder_type  = param.decoder
    encoder = param.encoder
    decoder = param.decoder
    replace_rate  = param.replace_rate
    optim_type    = param.optimizer 
    loss_fn       = param.loss_fn
    lr             = param.lr
    weight_decay   = param.weight_decay
    lr_f           = param.lr_f
    weight_decay_f = param.weight_decay_f
    linear_prob    = param.linear_prob
    load_model     = param.load_model
    save_model     = param.save_model
    logs           = param.logging
    use_scheduler  = param.scheduler
    batch_size     = param.batch_size
    batch_size_f     = param.batch_size_f
    param.num_features = num_features
    
    optimizer_type = param.optimizer

    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)


    # for i, seed in enumerate(seeds):
    #     print(f"####### Run {i+1} for seed {seed}")
    set_random_seed(0)

    if logs:
        logger = TBLogger(name=f"{dataset_name.lower()}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
    else:
        logger = None

    model_name = f"{encoder}_{decoder}_{num_hidden}_{num_layers}_{dataset_name}_{param.mask_rate}_{num_hidden}_checkpoint.pt"

    model = build_model(param)

    if not param.no_pretrain:
        # ------------- pretraining starts ----------------
        if not load_model:
            logging.info("---- start pretraining ----")
            model = pretrain_mini_batch(param, model, graph,optimizer=optimizer_type, batch_size=batch_size, lr=lr, weight_decay=weight_decay, max_epoch=max_epoch, device=device, use_scheduler=use_scheduler, model_dir=model_dir, model_name=model_name)
        
            model = model.cpu()
            logging.info(f"saving model to {model_dir}/{model_name}...")
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        # ------------- pretraining ends ----------------   

        if load_model:
            model.load_state_dict(torch.load(os.path.join(param.checkpoint_path)))
            logging.info(f"Loading Model from {param.checkpoint_path}...")
    else:
        logging.info("--- no pretrain ---")


    # eval
    model = model.to(device)
    model.eval()

    logging.info("---- start finetuning / evaluation ----")

    final_accs = []
    for i,_ in enumerate(seeds):
        print(f"############################### Run eval seed {seeds[i]} ###############################")
        set_random_seed(seeds[i])
        eval_model = build_model(param)
        eval_model.load_state_dict(model.state_dict())
        eval_model.to(device)

        print(f"features size : {graph.ndata['feat'].shape[1]}")
        logging.info("start evaluation...")

        final_acc = evaluete_mini_batch(
                        eval_model, 
                        graph, labels,
                        split_idx,
                        num_classes, 
                        lr_f, weight_decay_f, max_epoch_f, 
                        linear_prob, 
                        device, batch_size=batch_size_f, shuffle=True)

        final_accs.append(float(final_acc))

        print(f"Run {seeds[i]} | TestAcc: {final_acc:.4f}")
        
    print(f"# final_acc: {np.mean(final_accs):.4f}, std: {np.std(final_accs):.4f}")
    
    if logger is not None:
        logger.finish()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str,               default= 'Cora') #['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv','Reddit','Flickr','Yelp']
    parser.add_argument('--attack',         type=str,               default=  'no') # ['DICE-0.1','Meta_Self-0.05' ,...] 攻击方式-扰动率
    margs = parser.parse_args()
    Train_SPMGAE_nodecls(margs)