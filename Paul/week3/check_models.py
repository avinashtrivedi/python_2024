import pickle
from torchinfo import summary

def ensure_boe_model_is_correct(boe_model):
    model_summary = summary(boe_model)

    with open('../../assets/assignment3/model3a_summary.pkl', 'rb') as f:
        lst_solution = pickle.load(f)
    
    # lst_solution = list_solution[1:]
    lst_student = model_summary.summary_list[1:]
    
    num_emb_bag_layer = 0
    num_fc_layer = 0
    for layer in lst_student:
        if layer.class_name == 'EmbeddingBag':
            num_emb_bag_layer += 1
        if layer.class_name == 'Linear':
            num_fc_layer += 1
    
    assert num_emb_bag_layer == 1, "You should have exactly one EmbeddingBag layer"
    assert num_fc_layer == 1, "You should have exactly one fully connected layer"
    
    # now check the order
    if not lst_student[0].class_name == 'EmbeddingBag' and \
       not lst_student[1].class_name == 'Linear':
        assert False, "You should have 1 linear layer which follows the EmbeddingBag layer"
    
    # sanity check
    assert len(lst_solution) == len(lst_student), "Make sure that you have exactly 1 EmbeddingBag and 1 Linear layer"
    
    # check the parameters of the embedding bag layer
    emb_bag_layer = lst_student[0]
    # emb_bag_layer_solution = lst_solution[0]
    assert emb_bag_layer.module.num_embeddings == lst_solution[0]['num_embeddings'], "You must set num_embeddings to vocabulary size"
    assert emb_bag_layer.module.embedding_dim == 5, "You must set the embedding_dim to hyperparameter d"
    for layer_solution, layer_student in zip(lst_solution, lst_student):
        # assert layer_solution.class_name == layer_student.class_name, f"Layer {layer_solution.class_name} does not match"
        assert layer_solution['num_params'] == layer_student.num_params, f"Number of parameters in layer {layer_student.class_name} do not match"
        
def ensure_vec_model_is_correct(vec_model):
    model_summary = summary(vec_model)

    with open('../../assets/assignment3/model3b_summary.pkl', 'rb') as f:
        lst_solution = pickle.load(f)
    
    # lst_solution = m_summary_solution.summary_list[1:]
    lst_student = model_summary.summary_list[1:]
    
    num_fc_layer = 0
    for layer in lst_student:
        if layer.class_name == 'Linear':
            num_fc_layer += 1
            
    assert num_fc_layer == 2, "You should have exactly two fully connected layer"
    
    # sanity check
    assert len(lst_solution) == len(lst_student), "Make sure that you have exactly 2 Linear layers"
    
    for layer_solution, layer_student in zip(lst_solution, lst_student):
        # assert layer_solution.class_name == layer_student.class_name, f"Layer {layer_solution.class_name} does not match"
        assert layer_solution['num_params'] == layer_student.num_params, f"Number of parameters in layer {layer_student.class_name} do not match"
