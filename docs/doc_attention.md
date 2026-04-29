# Code Documentation: `marl_models/attention.py`

## 1. Overview

This module implements the **Graph Attention Network (GAT)** logic used by the attention-based MARL algorithms (Attention-MADDPG, Attention-MAPPO, Attention-MASAC, etc.).

## 2. Class: `CrossAttentionExtractor`

This is the core building block implementing **Scaled Dot-Product Attention**. It allows a "Query" entity (e.g., the UAV itself) to extract relevant information from a set of "Key/Value" entities (e.g., its neighbors).

### `__init__(self, self_dim, target_dim)`

Initializes the linear layers required for the attention mechanism.

* **Args:**
* `self_dim`: Dimension of the querying entity's features.
* `target_dim`: Dimension of the target entities' features.


* **Components:**
* `query_layer`: Projects the agent's state into Query space.
* `key_layer`, `value_layer`: Projects neighbor/UE states into Key and Value spaces.
* `out_proj`: Final linear projection for the aggregated context vector.

### `forward(self, self_embedding, target_embeddings, mask=None)`

Performs the forward pass of the attention mechanism using PyTorch's optimized scaled-dot-product attention.

1. **Projections:** Transforms inputs into Multi-Head Q, K, V matrices and reshapes them to `(Batch, Heads, Sequence, Head_Dim)`.
2. **Masking:** If a `mask` is provided (detecting padding), it is reshaped into a boolean mask to instruct the attention kernel to ignore padded entities.
3. **SDPA (Scaled Dot-Product Attention):** Uses PyTorch's highly optimized, memory-efficient `F.scaled_dot_product_attention` to compute similarities and weighted sums in a single fused operation.
4. **Aggregation:** Transposes and flattens the multi-head output back into a single context vector, passing it through the final `out_proj` layer.

## 3. Class: `AttentionActorBase`

This is a base class for the **Actor** (Policy) networks in algorithms like Attention-MADDPG and Attention-MAPPO. It replaces the standard MLP feature extractor.

### **Purpose**

To create a fixed-size embedding vector from a complex, dynamic observation that includes:

1. **Self State:** The UAV's own position, velocity, etc.
2. **Neighbors:** A list of relative positions of other UAVs.
3. **UEs:** A list of requests and positions from ground users.

### **Key Methods**

* **`__init__(self, obs_dim)`**:
* Defines distinct encoders (`nn.Linear` + `LayerNorm` + `ReLU`) for Self, Neighbors, and UEs.
* Initializes two `CrossAttentionExtractor` modules: one for Neighbors and one for UEs.
* Defines a `fusion` layer to combine the outputs.


* **`get_feature_embedding(self, obs_flat)`**:
* **Slicing:** Takes the flattened observation vector provided by the environment and slices it back into its constituent parts: `[Self | Neighbors | UEs]`.
* **Mask Generation:** Automatically detects "padding" (rows of zeros) in the neighbor/UE lists and creates a binary mask to tell the attention layer to ignore them.
* **Encoding & Attention:** Passes valid entities through encoders and then through the cross-attention layers.
* **Fusion:** Concatenates `[Self_Embedding, Neighbor_Context, UE_Context]` and passes them through a final MLP to produce the state representation.

## 4. Class: `AttentionCriticBase`

This is a base class for the **Centralized Critic** networks. It draws inspiration from **MAAC (Multi-Actor Attention-Critic)**.

### **Purpose**

In Multi-Agent RL, a centralized critic needs to evaluate the global state. Simply concatenating all agent states works poorly as the number of agents grows. This class uses attention to let the critic "focus" on the specific agents that are most relevant to the agent being evaluated.

### **Key Methods**

* **`get_all_embeddings(self, inputs)`**:
* Encodes the raw state/action pairs of all agents into a hidden representation.

* **`attend_to_others(self, embeddings, num_agents, agent_index)`**:
* **Logic (1-to-N Attention):** When evaluating Agent $i$, we treat Agent $i$ as the "Query" (`me_embedding`). All *other* agents are treated as "Keys/Values".
* Answers the question: *"How do the actions of other agents impact Agent $i$?"*
* Used sequentially by Off-Policy algorithms (MADDPG, MATD3, MASAC) where each agent has an independent critic network.

* **`vectorized_attend_to_others(self, embeddings)`**:
* **Logic (N-to-N Batched Attention):** Computes the attention context for the *entire swarm simultaneously* in $O(1)$ GPU operations.
* Uses a boolean mask (`~torch.eye`) to efficiently broadcast and prevent any agent from attending to itself.
* Crucial for centralized shared critics (like MAPPO) to avoid an $O(N)$ Python loop bottleneck during the forward pass.

* **`get_q_embedding(self, obs_tensor, action_tensor, agent_index)`**:
* A helper method for Q-learning models that concatenates Observations and Actions, encodes them, runs `attend_to_others`, and returns the final fused embedding.
