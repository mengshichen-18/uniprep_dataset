# 请你帮我写一下python代码，从./datalake途径下面，读取所有的csv的文件，然后构建图中的hierarcical graph (从底到顶，分别是token layer, cell layer, row/column layer, table layer)，包括node_id_mapping(每一个token, cell, row, column, table_name对应的id)，和他们之间的edge_list。包括两块，第一块是不同layer之间的node连接关系，比如一个cell之中对应的所有的token node，都应该和这个cell nod有一条边；一个row/column node之中对应的所有的cell node，都应该和这个row/column node有一条边; 一个table node之中对应的所有的row/column node，都应该和这个table node有一条边。第二块是同一个layer之内的如果有关系的话，应该要有一条边连接起来（这个关系从./label/entity_matching/entity_matching_labels.csv中读取）. 注意，这里哟个的tokenizer是从"/data1/jianweiw/LLM/models_hf/sentence-t5-base"这里获得的

import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoTokenizer = None

DEFAULT_TOKENIZER_PATH = "/home/mengshi/embedding_model/models_hf/sentence-t5-base"
DEFAULT_DATALAKE_PATH = "/home/mengshi/table_quality/datasets_joint_discovery_integration/magellan_1218/datalake_plus"
DEFAULT_LABEL_PATH = "/home/mengshi/table_quality/datasets_joint_discovery_integration/magellan_1218/label_plus"
DEFAULT_OUTPUT_DIR = "./magellan_no_token"
DEFAULT_MAX_CELL_LENGTH = 500

def find_long_cells(df, max_length=100):
    long_cells = []
    for row_idx, row in df.iterrows():
        for col in df.columns:
            value = row[col]
            if isinstance(value, str) and len(value) > max_length:
                long_cells.append({
                    'row': row_idx,
                    'column': col,
                    'length': len(value),
                    'content': value[:50] + '...' if len(value) > 50 else value  # 预览前 50 个字符
                })
    return pd.DataFrame(long_cells)

def truncate_long_cells(df, max_length=100):
    def truncate(x):
        if isinstance(x, str) and len(x) > max_length:
            return x[:max_length]  # 直接截断
        return x  # 非字符串或长度正常则原样返回
    
    return df.map(truncate)

class HierarchicalGraphBuilder:
    # santos_benchmark, magellan, wikidbs
    def __init__(
        self,
        datalake_path: str = DEFAULT_DATALAKE_PATH,
        label_path: str = DEFAULT_LABEL_PATH,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        random_seed: int = 42,
        max_cell_length: int = DEFAULT_MAX_CELL_LENGTH,
    ):
        self.datalake_path = datalake_path
        self.label_path = label_path
        if AutoTokenizer is None:
            self.tokenizer = None
            print("[WARN] 'transformers' not installed; tokenizer disabled (not used by graph builder).")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.random_seed = random_seed
        self.max_cell_length = max_cell_length
        np.random.seed(random_seed)
    

        self.node_id_mapping = {
            'cell': {},      # (cell_name) -> node_id
            'row': {},       # (table_name, row_idx) -> node_id
            'column': {},    # (table_name, col_idx) -> node_id
            'table': {}      # table_name -> node_id
        }
        

        self.edge_lists = []
        self.edge_labels = []
        self.edge_type = []

        self.train_mask = []
        self.validate_mask = []
        self.test_mask = []
        
        # Counter for unique node IDs
        self.node_counter = -1
        
        # Store table data for processing
        self.table_data = {}
        self.row_id_lookup = {}
        
    def get_next_node_id(self):
        """Get next unique node ID"""
        self.node_counter += 1
        return self.node_counter

    @staticmethod
    def _to_int(value):
        try:
            return int(value)
        except Exception:
            return int(float(value))

    def _resolve_em_row_key(self, table_name: str, row_id: int):
        direct_key = (table_name, row_id)
        if direct_key in self.node_id_mapping['row']:
            return direct_key
        lookup = self.row_id_lookup.get(table_name, {})
        if row_id in lookup:
            return (table_name, lookup[row_id])
        return direct_key
    
    def load_csv_files(self):
        """Load all CSV files from datalake directory"""
        csv_files = []
        for root, dirs, files in os.walk(self.datalake_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Use relative path as table name
                table_name = os.path.relpath(csv_file, self.datalake_path)
                table_name = table_name.replace('\\', '/').replace('.csv', '')
                
                df = pd.read_csv(csv_file)
                print(f"Loaded table: {table_name}, shape: {df.shape}")
                
                result = find_long_cells(df, max_length=self.max_cell_length)
                if result.shape[0] > 0:
                    print(f"Long cells found in {table_name}:")
                    print(result)
                df = truncate_long_cells(df, max_length=self.max_cell_length)
                df = df.fillna('UNKNOWN_CELL')

                self.table_data[table_name] = df
                if "id" in df.columns:
                    lookup = {}
                    for row_idx, raw_id in enumerate(df["id"].tolist()):
                        try:
                            lookup[self._to_int(raw_id)] = row_idx
                        except Exception:
                            continue
                    self.row_id_lookup[table_name] = lookup
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    def build_table_layer(self):
        """Build table layer nodes"""
        print("Building table layer...")
        for table_name in self.table_data.keys():
            if table_name not in self.node_id_mapping['table']:
                self.node_id_mapping['table'][table_name] = self.get_next_node_id()
    
    def build_row_column_layer(self):
        """Build row and column layer nodes"""
        print("Building row and column layers...")
        for table_name, df in self.table_data.items():
            # Build row nodes
            for row_idx in range(len(df)):
                row_key = (table_name, row_idx)
                if row_key not in self.node_id_mapping['row']:
                    self.node_id_mapping['row'][row_key] = self.get_next_node_id()
            
            # Build column nodes
            for col_idx in range(len(df.columns)):
                col_key = (table_name, df.columns[col_idx])
                if col_key not in self.node_id_mapping['column']:
                    self.node_id_mapping['column'][col_key] = self.get_next_node_id()
    
    def build_cell_layer(self):
        """Build cell layer nodes"""
        print("Building cell layer...")
        for table_name, df in self.table_data.items():
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    # cell_key = (table_name, row_idx, col_idx)
                    cell_key = df.iloc[row_idx, col_idx]
                    if cell_key not in self.node_id_mapping['cell']:
                        self.node_id_mapping['cell'][cell_key] = self.get_next_node_id()
    
    def build_hierarchical_edges(self):
        """Build edges between different layers"""
        print("Building hierarchical edges...")
        
        print("Building cell -> row and cell -> column edges...")
        # Build cell -> row and cell -> column edges
        # for table_name, df in self.table_data.items():
        for table_name, df in tqdm(self.table_data.items(), desc="Processing tables"):
            for row_idx in range(len(df)):
                for col_idx in range(len(df.columns)):
                    cell_key = df.iloc[row_idx, col_idx]
                    cell_id = self.node_id_mapping['cell'][cell_key]
                    
                    # Cell -> row edge with training indicator
                    # row_key = (table_name, row_idx)
                    row_key = (table_name, row_idx)
                    row_id = self.node_id_mapping['row'][row_key]
                    # self.edge_lists['cell_row'].append((cell_id, row_id))
                    self.edge_lists.append((cell_id, row_id))
                    self.edge_type.append('cell_row')
                    self.train_mask.append(1)
                    self.validate_mask.append(0)
                    self.test_mask.append(0)
                    self.edge_labels.append(1)
                    
                    # Cell -> column edge with training indicator
                    col_key = (table_name, df.columns[col_idx])
                    col_id = self.node_id_mapping['column'][col_key]
                    # self.edge_lists['cell_column'].append((cell_id, col_id))
                    self.edge_lists.append((cell_id, col_id))
                    self.edge_type.append('cell_column')
                    self.train_mask.append(1)
                    self.validate_mask.append(0)
                    self.test_mask.append(0)
                    self.edge_labels.append(1)
        
        print("Building row -> table and column -> table edges...")
        # Build row -> table and column -> table edges
        # for table_name, df in self.table_data.items():
        for table_name, df in tqdm(self.table_data.items(), desc="Processing tables"):
            table_id = self.node_id_mapping['table'][table_name]
            
            # Row -> table edges with training indicator
            for row_idx in range(len(df)):
                # row_key = (table_name, row_idx)
                row_key = (table_name, row_idx)
                row_id = self.node_id_mapping['row'][row_key]
                # self.edge_lists['row_table'].append((row_id, table_id))
                self.edge_lists.append((row_id, table_id))
                self.edge_type.append('row_table')
                self.train_mask.append(1)
                self.validate_mask.append(0)
                self.test_mask.append(0)
                self.edge_labels.append(1)
            
            # Column -> table edges with training indicator
            for col_idx in range(len(df.columns)):
                col_key = (table_name, df.columns[col_idx])
                col_id = self.node_id_mapping['column'][col_key]
                # self.edge_lists['column_table'].append((col_id, table_id))
                self.edge_lists.append((col_id, table_id))
                self.edge_type.append('column_table')
                self.train_mask.append(1)
                self.validate_mask.append(0)
                self.test_mask.append(0)
                self.edge_labels.append(1)
    
    def build_task_edges(self):
        """Build edges based on entity matching labels"""
        print("Building entity matching edges...")
        
        tasks = ["entity_matching", "joinable_table_search", "schema_matching", "unionable_table_search"]


        for task in tasks:
            print(f"Loading {task} labels...")
            for file_name in ["train.csv", "validate.csv", "test.csv"]:
                file_path = os.path.join(self.label_path, task, file_name)
                label_df = pd.read_csv(file_path)
                print(f"Loaded {task} labels: {label_df.shape}")
                
                for _, row in label_df.iterrows():
                    edge_added = False
                    #  Add matching edges based on the type of matching
                    if task == "entity_matching":
                        table1 = row['ltable_name'].replace('\\', '/').replace('.csv', '')
                        table2 = row['rtable_name'].replace('\\', '/').replace('.csv', '')
                        l_id = self._to_int(row['l_id'])
                        r_id = self._to_int(row['r_id'])
                        row1_key = self._resolve_em_row_key(table1, l_id)
                        row2_key = self._resolve_em_row_key(table2, r_id)

                        if row1_key in self.node_id_mapping['row'] and row2_key in self.node_id_mapping['row']:
                            id1 = self.node_id_mapping['row'][row1_key]
                            id2 = self.node_id_mapping['row'][row2_key]

                            # self.edge_lists['entity_matching'].append((id1, id2))
                            self.edge_lists.append((id1, id2))
                            self.edge_type.append('entity_matching')
                            self.edge_labels.append(int(row['label']))
                            edge_added = True

                        else:
                            print(f"Row keys not found: {row1_key}, {row2_key}")
                            continue
                    
                    elif task == "joinable_table_search":
                        table1 = row['table_name_1'].replace('\\', '/').replace('.csv', '')
                        table2 = row['table_name_2'].replace('\\', '/').replace('.csv', '')
                        col1_key = (table1, row['column_name_1'])
                        col2_key = (table2, row['column_name_2'])
                        if table1 not in self.node_id_mapping['table'] or table2 not in self.node_id_mapping['table']:
                            print(f"joinable_table_search table not found: {table1}, {table2}")
                            continue
                        table1_id = self.node_id_mapping['table'][table1]
                        table2_id = self.node_id_mapping['table'][table2]

                        if col1_key in self.node_id_mapping['column'] and col2_key in self.node_id_mapping['column']:
                            id1 = self.node_id_mapping['column'][col1_key]
                            id2 = self.node_id_mapping['column'][col2_key]

                            # self.edge_lists['joinable_table_search'].append((id1, id2))
                            self.edge_lists.append((id1, id2, table1_id, table2_id))
                            self.edge_type.append('joinable_table_search')
                            self.edge_labels.append(int(row['label']))
                            edge_added = True

                        else:
                            print(f"Column keys not found: {col1_key}, {col2_key}")
                            continue


                    elif task == "schema_matching":
                        table1 = row['table_name_1'].replace('\\', '/').replace('.csv', '')
                        table2 = row['table_name_2'].replace('\\', '/').replace('.csv', '')
                        col1_key = (table1, row['renamed_column_name_1'])
                        col2_key = (table2, row['renamed_column_name_2'])

                        if col1_key in self.node_id_mapping['column'] and col2_key in self.node_id_mapping['column']:
                            id1 = self.node_id_mapping['column'][col1_key]
                            id2 = self.node_id_mapping['column'][col2_key]

                            # self.edge_lists['entity_matching'].append((id1, id2))
                            self.edge_lists.append((id1, id2))
                            self.edge_type.append('schema_matching')
                            self.edge_labels.append(int(row['label']))
                            edge_added = True

                        else:
                            print(f"schema_matching keys not found: {col1_key}, {col2_key}")
                            continue
                    

                    elif task == "unionable_table_search":
                        table1 = row['table_name_1'].replace('\\', '/').replace('.csv', '')
                        table2 = row['table_name_2'].replace('\\', '/').replace('.csv', '')

                        if table1 in self.node_id_mapping['table'] and table2 in self.node_id_mapping['table']:
                            id1 = self.node_id_mapping['table'][table1]
                            id2 = self.node_id_mapping['table'][table2]

                            # self.edge_lists['entity_matching'].append((id1, id2))
                            self.edge_lists.append((id1, id2))
                            self.edge_type.append('union_table_search')
                            self.edge_labels.append(int(row['label']))
                            edge_added = True

                        else:
                            print(f"union_table_search keys not found: {table1}, {table2}")
                            continue
                    
                    else:
                        print(f"Unknown task: {task}")
                        continue

                    if not edge_added:
                        continue
                    
                    if file_name == "train.csv":
                        self.train_mask.append(1)
                        self.validate_mask.append(0)
                        self.test_mask.append(0)
                    elif file_name == "validate.csv":
                        self.train_mask.append(0)
                        self.validate_mask.append(1)
                        self.test_mask.append(0)
                    elif file_name == "test.csv":
                        self.train_mask.append(0)
                        self.validate_mask.append(0)
                        self.test_mask.append(1)

                    else:
                        print(f"Unknown file name: {file_name}")
                        continue
    
    
    def build_graph(self):
        """Build the complete hierarchical graph"""
        print("Starting to build hierarchical graph...")
        
        # Load CSV files
        self.load_csv_files()
        
        # Build layers from top to bottom
        self.build_table_layer()
        self.build_row_column_layer()
        self.build_cell_layer()
        
        # Build edges
        self.build_hierarchical_edges()
        self.build_task_edges()
        
        print("Graph building completed!")
        self.print_statistics()
    
    def print_statistics(self):
        """Print graph statistics"""
        print("\n=== Graph Statistics ===")
        print(f"Total nodes: {self.node_counter}")
        print(f"Table nodes: {len(self.node_id_mapping['table'])}")
        print(f"Row nodes: {len(self.node_id_mapping['row'])}")
        print(f"Column nodes: {len(self.node_id_mapping['column'])}")
        print(f"Cell nodes: {len(self.node_id_mapping['cell'])}")
        
        print(f"\nEdge counts: {len(self.edge_lists)}")
    
    # santos_benchmark, magellan, wikidbs
    def save_graph(self, output_path: str = DEFAULT_OUTPUT_DIR):
        """Save the graph to files"""
        os.makedirs(output_path, exist_ok=True)
        
        # Save node mappings
        with open(os.path.join(output_path, 'node_id_mapping.json'), 'w') as f:
            # Convert tuple keys to strings for JSON serialization
            serializable_mapping = {}
            for layer, mapping in self.node_id_mapping.items():
                serializable_mapping[layer] = {}
                for key, value in mapping.items():
                    serializable_mapping[layer][str(key)] = value
            json.dump(serializable_mapping, f, indent=2)
        
        # 保存 edge_lists
        with open(f'{output_path}/edge_lists.txt', 'w') as f:
            for edge in self.edge_lists:
                if len(edge) == 2:
                    f.write(f'{edge[0]}\t{edge[1]}\n')  # 用 tab 分隔，便于后续读取
                elif len(edge) == 4:
                    f.write(f'{edge[0]}\t{edge[1]}\t{edge[2]}\t{edge[3]}\n')
                else:
                    print(f"Invalid edge: {edge}")

        # 保存 edge_labels
        with open(f'{output_path}/edge_labels.txt', 'w') as f:
            for label in self.edge_labels:
                f.write(f'{label}\n')

        # 保存 edge_type
        with open(f'{output_path}/edge_type.txt', 'w') as f:
            for e_type in self.edge_type:
                f.write(f'{e_type}\n')

        # 保存 train_mask
        with open(f'{output_path}/train_mask.txt', 'w') as f:
            for mask in self.train_mask:
                f.write(f'{mask}\n')

        # 保存 validate_mask
        with open(f'{output_path}/validate_mask.txt', 'w') as f:
            for mask in self.validate_mask:
                f.write(f'{mask}\n')

        # 保存 test_mask
        with open(f'{output_path}/test_mask.txt', 'w') as f:
            for mask in self.test_mask:
                f.write(f'{mask}\n')

        print(f"Graph saved to {output_path}")
    


def parse_args():
    parser = argparse.ArgumentParser(description="Build hierarchical graph (no-token pipeline).")
    parser.add_argument("--datalake-path", default=DEFAULT_DATALAKE_PATH, help="Path to datalake CSV directory.")
    parser.add_argument("--label-path", default=DEFAULT_LABEL_PATH, help="Path to label directory.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for graph files (node_id_mapping, edge files, masks).",
    )
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH, help="HuggingFace tokenizer/model path.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-cell-length", type=int, default=DEFAULT_MAX_CELL_LENGTH, help="Truncate cell strings to this length.")
    return parser.parse_args()


# Usage example
if __name__ == "__main__":
    args = parse_args()
    # Initialize the graph builder
    builder = HierarchicalGraphBuilder(
        datalake_path=args.datalake_path,
        label_path=args.label_path,
        tokenizer_path=args.tokenizer_path,
        random_seed=args.random_seed,
        max_cell_length=args.max_cell_length,
    )
    
    # Build the graph
    builder.build_graph()
    
    # Save the graph
    builder.save_graph(output_path=args.output_dir)
    
    # Access the results
    print("\nAccessing results:")
    print("Node ID mappings available in: builder.node_id_mapping")
    print("Edge lists available in: builder.edge_lists")
    
    # Example: Get all edges as a single list with training indicators
    # all_edges = []
    # for edge_type, edges in builder.edge_lists.items():
    #     for edge in edges:
    #         all_edges.append((edge['source'], edge['target'], edge['training_indicator'], edge_type))
    
    # print(f"Total edges: {len(all_edges)}")
    
    # # Example: Count training vs testing edges
    # train_edges = sum(1 for edge_type, edges in builder.edge_lists.items() 
    #                  for edge in edges if edge['training_indicator'] == 1)
    # test_edges = sum(1 for edge_type, edges in builder.edge_lists.items() 
    #                 for edge in edges if edge['training_indicator'] == 0)
    
    # print(f"Training edges: {train_edges}")
    # print(f"Testing edges: {test_edges}")
    # print(f"Training ratio: {train_edges / (train_edges + test_edges):.2f}")
