
class PointerNetwork_v3(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, config, encoder):
        """Initialize params."""
        super(PointerNetwork_v3, self).__init__()
        self.config  = config
        self.encoder = encoder

        self.input_size   = self.config.hidden_dim
        self.hidden_size  = self.config.hidden_dim
        self.context_size = self.config.hidden_dim

        self.input_weights_1 = nn.Parameter(
            torch.Tensor(4 * hidden_size, input_size)
        )
        self.hidden_weights_1 = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size)
        )
        self.input_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.hidden_bias_1 = nn.Parameter(torch.Tensor(4 * hidden_size))


        self.context2attention = nn.Parameter(
            torch.Tensor(context_size, context_size)
        )
        self.bias_context2attention = nn.Parameter(torch.Tensor(context_size))

        self.hidden2attention = nn.Parameter(
            torch.Tensor(context_size, hidden_size)
        )

        self.input2attention = nn.Parameter(
            torch.Tensor(input_size, context_size)
        )

        self.recurrent2attention = nn.Parameter(torch.Tensor(context_size, 1))
        self.stop_vector = nn.Parameter(torch.randn(1, 1, self.input_size))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        stdv_ctx = 1.0 / math.sqrt(self.context_size)
        stdv_stop = 1.0 / math.sqrt(self.input_size)

        self.input_weights_1.data.uniform_(-stdv, stdv)
        self.hidden_weights_1.data.uniform_(-stdv, stdv)
        self.input_bias_1.data.fill_(0)
        self.hidden_bias_1.data.fill_(0)


        self.context2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.bias_context2attention.data.fill_(0)

        self.hidden2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.input2attention.data.uniform_(-stdv_ctx, stdv_ctx)

        self.recurrent2attention.data.uniform_(-stdv_ctx, stdv_ctx)
        self.stop_vector.data.uniform_(-stdv_stop, stdv_stop)


    def forward(self, sentence, sentence_lens, char_ids, char_lens, pointer_answers):
        """Propogate input through the network."""
        def recurrence(input, hidden, projected_ctx):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim

            gates = F.linear(
                input, self.input_weights_1, self.input_bias_1
            ) + F.linear(hx, self.hidden_weights_1, self.hidden_bias_1)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim

            # Attention mechanism

            # Project current hidden state to context size
            hidden_ctx = F.linear(hy, self.hidden2attention)

            # Added projected hidden state to each projected context
            hidden_ctx_sum = projected_ctx + hidden_ctx.unsqueeze(0).expand(
                projected_ctx.size()
            )

            # Non-linearity
            hidden_ctx_sum = F.tanh(hidden_ctx_sum)

            # Compute alignments
            alpha = torch.bmm(
                hidden_ctx_sum.transpose(0, 1),
                self.recurrent2attention.unsqueeze(0).expand(
                    hidden_ctx_sum.size(1),
                    self.recurrent2attention.size(0),
                    self.recurrent2attention.size(1)
                )
            ).squeeze()

            return (hy, cy), alpha


        batch_size, sent_size = sentence.size()
        pointer_answers = pointer_answers.transpose(1, 0).contiguous() # (k , N)

        context, h = self.encoder(sentence, char_ids, char_lens)
        context = context.permute(1, 0, 2)  #(W, N, input_size)
        ctx     = torch.cat((context, self.stop_vector.repeat(1, batch_size, 1)), dim = 0) #(W+1, N, input_size)

        hidden = h, h # (N, hidden_size)


        projected_ctx = torch.bmm(
            ctx,
            self.context2attention.unsqueeze(0).expand(
                ctx.size(0),
                self.context2attention.size(0),
                self.context2attention.size(1)
            ),
        )
        projected_ctx += \
            self.bias_context2attention.unsqueeze(0).unsqueeze(0).expand(
                projected_ctx.size()
            )


        output = []
        steps = range(pointer_answers.size(0))
        for i in steps:
            if i > 1:
                _, pointer_idx = output[-1].max(dim = -1)
                curr_inputs = ctx.permute(1,0,2).gather(1, pointer_idx.view(batch_size, 1, 1).repeat(1, 1, self.input_size)).squeeze(1) #(N, input_size)
            else:
                curr_inputs = self.init_hidden(batch_size, self.input_size). #(N, input_size)

            hidden, alphas = recurrence(curr_inputs, hidden, projected_ctx)
            output.append(alphas)

        output = torch.cat(output, 0).view(-1, batch_size, sent_size+1).permute(1,0,2).contiguous().view(-1, sent_size+1)
        return output

