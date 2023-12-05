
from     easydict        import EasyDict
import   torch
from     utils          import *
from     build_easydict import *
from     evaluation     import * 
from     datasets_dgl.data_dgl import *

import   logging
from     tqdm import tqdm
from     models import build_model
import torch.nn.functional as F
from utils import compute_edge_sim

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)



def pretrain_tranductive(param, model, graph, feat, label, train_mask, val_mask, test_mask, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None):
    logging.info("start training SPMGAE node classification..")
    graph = graph.to(device)
    x = feat.to(device)

    ###########################################################
    n_node = graph.num_nodes()
    
    # homophily_score_before = compute_graph_homophily_score(graph, label)
    # print(homophily_score_before)

    edges_sim       = compute_edge_sim(graph.edges(), x, sim_mode=param.sim_mode)
    
    ref_edges_index = torch.where(edges_sim >= param.keep_threshold)
    del_edges_index = torch.where(edges_sim <  param.dele_threshold)

    edges           = torch.stack((graph.edges()[0],graph.edges()[1]),dim=0)
    ref_edges       = edges[:, ref_edges_index[0]]
    del_edges       = edges[:, del_edges_index[0]]
    graph_refine    = dgl.graph((ref_edges[0],ref_edges[1]), num_nodes=n_node).to(device)
    graph_refine = graph_refine.remove_self_loop()
    graph_refine = graph_refine.add_self_loop()


    # homophily_score_after = compute_graph_homophily_score(graph_refine, label)
    # print(homophily_score_after)

    print('num  edges : {}'.format(graph.num_edges()))
    print('keep edges : {}'.format(len(ref_edges_index[0])))
    print('del  edges : {}'.format(len(del_edges_index[0])))

    ############################################################

    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()

        # loss, loss_dict = model(graph, x, epoch)
        loss, loss_dict = model(graph_refine, del_edges, x, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        if (epoch + 1) % 200 == 0:
            node_classification_evaluation(model, graph, x, label, train_mask, val_mask, test_mask, num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob, mute=True)

    # return best_model
    return model



def Train_SPMGAE_nodecls(margs):
    #########################
    device = f"cuda" if torch.cuda.is_available() else "cpu"
  
    dataset_name = margs.dataset
    if dataset_name.split('-')[0] == 'Attack':
        # dataset_name = dataset_name.split('-')[1]
        DATASET = EasyDict()
        DATASET.ATTACK = {
            "data":dataset_name,
            "attack":margs.attack.split('-')[0],
            "ptb_rate":margs.attack.split('-')[1]
        }
        # now just attack use
        dataset  = load_attack_data(DATASET['ATTACK'])
        graph = dataset.graph    
    else:
        dataset  = load_data(dataset_name)
        graph = dataset[0]

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
        param = load_best_configs(param, dataset_name.split('-')[1].lower() if dataset_name.split('-')[0] == 'Attack' else dataset_name.lower() , "./configs.yml")


    # ##################################################################################################################
    # # grid search
    # grid_num = 0 
    # # for gamma in [0, 1, 10]:
    # #     for beta in [0.0, 0.001, 0.1]:
    # for decay in [0.2, 0.8]:
    # #           for add_rate in [0.2, 0.4, 0.6, 0.8]:
    #     for threshold in [[0.04, 0.02], [0.04, 0.04],[0.06,0.02], [0.06,0.04],[0.08, 0.02],[0.08, 0.04],[0.1, 0.02],[0.1, 0.04]]:
    #         for type_graph4recon in ['refine', 'ptb']:
    #             for num_hidden in [256, 512, 1024]:
    #                 grid_num += 1
    #                 # param.gamma              = gamma
    #                 # param.beta               = beta
    #                 param.decay              = decay
    #                 # param.add_rate           = add_rate
    #                 param.num_hidden         = num_hidden
    #                 param.type_graph4recon   = type_graph4recon
    #                 param.keep_threshold     = threshold[0]
    #                 param.dele_threshold     = threshold[1]
    #                 print('############################################################################')
    #                 print('grid search at experiment {}'.format(grid_num))
    #                 print('************************************************')
    #                 print('gamma                : {}'.format(param.gamma))
    #                 print('beta                 : {}'.format(param.beta))
    #                 print('add_rate             : {}'.format(param.add_rate))
    #                 print('decay                : {}'.format(param.decay))
    #                 print('num_hidden           : {}'.format(param.num_hidden))
    #                 print('keep_threshold       : {}'.format(param.keep_threshold))
    #                 print('dele_threshold       : {}'.format(param.dele_threshold))
    #                 print('type_graph4recon     : {}'.format(param.type_graph4recon))
    #                 print('************************************************')
    #                 ###############################################################################################################

    seeds         = param.seeds
    max_epoch     = param.max_epoch
    max_epoch_f   = param.max_epoch_f
    num_hidden    = param.num_hidden
    num_layers    = param.num_layers
    encoder_type  = param.encoder
    decoder_type  = param.decoder
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
    param.num_features = num_features


    # 
    feat       = graph.ndata["feat"]
    train_mask = graph.ndata["train_mask"].to(device)
    val_mask   = graph.ndata["val_mask"].to(device)
    test_mask  = graph.ndata["test_mask"].to(device)
    labels     = graph.ndata["label"].to(device)

    if param.use_complementary_graph:
        graph = get_complementary_graph(graph, feat)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph).to(device)


    acc_list = []
    estp_acc_list = []


    for i, seed in enumerate(seeds):
        print(f"####### Run {i+1} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name.lower()}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(param)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
            # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                    # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        

        if not load_model:
            model = pretrain_tranductive(param, model, graph, feat, labels, train_mask, val_mask, test_mask, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger)
        

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        final_acc, estp_acc = node_classification_evaluation(model, graph, feat, labels, train_mask, val_mask, test_mask,num_classes, lr_f, weight_decay_f, max_epoch_f, device, linear_prob)


        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',        type=str,               default= 'Cora') #['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv','Reddit','Flickr','Yelp']
    parser.add_argument('--attack',         type=str,               default=  'no') # ['DICE-0.1','Meta_Self-0.05' ,...] 攻击方式-扰动率
    margs = parser.parse_args()
    Train_SPMGAE_nodecls(margs)