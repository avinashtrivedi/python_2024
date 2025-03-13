import pickle
from torchinfo import summary

def ensure_model_is_correct(lstm_model):
    model_summary = summary(lstm_model)

    with open('../../assets/assignment4/model_summary.pkl', 'rb') as f:
        lst_solution = pickle.load(f)
    
    # lst_solution = m_summary_solution.summary_list[1:]
    lst_student = model_summary.summary_list[1:]
    
    num_lstm_layers = 0
    num_fc_layer = 0
    module_dict_used = False
    num_output_layers = 0
    for layer in lst_student:
        if layer.class_name == 'LSTM':
            num_lstm_layers += 1
        if layer.class_name == 'ModuleDict':
            module_dict_used = True
        if layer.class_name == 'Linear':
            if layer.depth == 1:
                num_fc_layer += 1
            elif layer.depth == 2:
                num_output_layers += 1
            
    if num_lstm_layers == 0:
        assert num_lstm_layers != 1, "You should have exactly one LSTM layer"
    if num_lstm_layers == 2:
        assert num_lstm_layers != 1, "You should have one LSTM layer, fyi you can set num_layers to 2 which will give you 2 stacked LSTMs"
    assert module_dict_used,  "To pass autograder, you must use ModuleDict; this ModuleDict should contain 7 linear layers"
    assert num_fc_layer == 1, "You should have exactly one fully connected layer"
    assert num_output_layers == 7, "To pass autograder, the ModuleDict should contain 7 linear layers"
    
    # now check the order
    if not lst_student[0].class_name == 'LSTM' and \
       not lst_student[1].class_name == 'Linear' and \
       not lst_student[2].class_name == 'ModuleDict':
        assert False, "You should have 1 linear layer which follows the LSTM layer and precedes the ModuleDict"
    
    # sanity check
    assert len(lst_solution) == len(lst_student), "Great job! You have used ModuleDict with 7 output layers. Make sure that the ModuleDict is the last layer"
    
    # check the parameters of the LSTM layer
    lstm_layer = lst_student[0]
    assert lstm_layer.module.num_layers == 2, "You must set num_layers to 2 which will give you 2 stacked LSTMs"
    assert lstm_layer.module.batch_first == True, "You must set batch_first to True to pass autograder"
    assert lstm_layer.module.hidden_size == 64, "You must set the hidden size to 64 to pass autograder"
    assert lstm_layer.module.bidirectional == False, "You must set bidirectional to False to pass autograder"
    
    for layer_solution, layer_student in zip(lst_solution, lst_student):
        # assert layer_solution.class_name == layer_student.class_name, f"Layer {layer_solution.class_name} does not match"
        assert layer_solution['num_params'] == layer_student.num_params, f"Number of parameters in layer {layer_student.class_name} do not match"
