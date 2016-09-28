package dp_process

/* 实现一个列(MLP)
 * 
 * |                 |       MLP(k-1)    |              MLP(k-1)                      |    MLP(k-1)      | 
 * |                 |       t=k-1       |              t=k                           |    t=k+1         |
 * |                 |                   |                                            |                  |
 * | lr_layer----------------------------------logisticregression_layer(nout个神经元)    |                  |                 
 * |                 |                   |              /\                            |                  |
 * |                 |                   |              ||w_lr                        |                  |
 * |                 |                   |              ||b_lr                        |                  |
 * |                 |                   |           ... ...                          |                  | 
 * |                 |           w_hidden2hidden[i]     ||                       w_hidden2hidden[i]      |
 * | hidden_layer[i]------------==================> hidden_layer[i](nhidden个神经元)==================>     |
 * |                 |           b_hidden2hidden[i]     /\                       b_hidden2hidden[i]      |
 * |                 |                   |              ||w_hidden[i]                 |                  |
 * |                 |                   |              ||b_hidden[i]                 |                  |
 * |                 |                   |           ... ...                          |                  |
 * |                 |           w_hidden2hidden[0]     ||                       w_hidden2hidden[0]      |
 * | hidden_layer[0]------------==================> hidden_layer[0](nhidden个神经元)==================>     |
 * |                 |           b_hidden2hidden[0]     /\                       b_hidden2hidden[0]      |
 * |                 |                   |              ||w_hidden[0]                 |                  |
 * |                 |                   |              ||b_hidden[0]                 |                  |
 * |                 |                   |          input(nin个神经元)                   |                  |
 * */

object RNN_MLP {
  
}