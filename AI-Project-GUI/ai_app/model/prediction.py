import torch
from .char_encoding import reverse_full_char_encoding

def create_expression(
    input_seq,
    max_expression_length,
    model,
    label_eos_token,
    label_sos_token,
):
  model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  with torch.no_grad():
    encoder_outputs, hidden, cell = model.encoder(input_seq)
    next_input = label_sos_token
    next_input = next_input[None,:]
    predicted_expression = [1]
    for _ in range(max_expression_length):
      next_input = next_input.unsqueeze(0)
      output, hidden, cell = model.decoder(next_input, encoder_outputs, hidden, cell)
      top1 = output.argmax(1)
      one_hot_encoded_top1 = torch.tensor([
        1 if i == top1.item()
        else 0
        for i in range(63)
      ])
      predicted_expression.append(top1.item())
      next_input = torch.tensor(one_hot_encoded_top1).float().to(device)
      next_input = next_input.unsqueeze(0)
      if next_input[0][2] == 1 or next_input[0][0] == 1:
        break
    return predicted_expression

def test_fn(model, test_data_loader):
    predicted_expression = []
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(len(test_data_loader))
    with torch.no_grad():
        for j, (src) in enumerate(test_data_loader):
          src = torch.transpose(src,0,1)
          src = src.to(device)
          label_sos_token = torch.tensor([1 if i == 1 else 0 for i in range(63)]).to(device)
          label_eos_token = torch.tensor([1 if i == 2 else 0 for i in range(63)]).to(device)
          predicted_expression = create_expression(src,200,model,label_sos_token,label_eos_token)
        predicted_expression_chars = [reverse_full_char_encoding[i-3] for i in predicted_expression if i > 2]
        print(predicted_expression_chars)
        predicted_expression_chars = "".join(predicted_expression_chars)

    return predicted_expression_chars