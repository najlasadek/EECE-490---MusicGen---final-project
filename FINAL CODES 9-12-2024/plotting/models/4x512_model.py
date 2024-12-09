import graphviz

digraph MusicLSTM {
   rankdir=LR;
   splines=true;
   bgcolor="#F0EEE6";
   node [shape=circle, style=filled, fontname="Helvetica-Bold", fontsize=12];
   
   // Input Layer (88 features)
   subgraph cluster_input {
       label=<<B>Input Layer</B>
(88 features)>;
       style=dashed;
       color="#004AAD";
       node [style=filled, color="#C7E1FF"];
       input1 [label=<<B>1</B>>];
       input2 [label=<<B>2</B>>];
       input3 [label=<<B><FONT POINT-SIZE="20">⋯</FONT></B>>];
       input88 [label=<<B>88</B>>];
   }

   // LSTM Layers with darker orange gradient
   subgraph cluster_lstm1 {
       label=<<B>LSTM Layer-1</B>
(512 units)>;
       style=solid;
       color="#FFA366";
       node [style=filled, color="#FFB380"];
       lstm1_1 [label=<<B>1</B>>];
       lstm1_2 [label=<<B>2</B>>];
       lstm1_3 [label=<<B><FONT POINT-SIZE="20">⋯</FONT></B>>];
       lstm1_512 [label=<<B>512</B>>];
   }

   subgraph cluster_lstm2 {
       label=<<B>LSTM Layer-2</B>
(512 units)>;
       style=solid;
       color="#FF884D";
       node [style=filled, color="#FF9966"];
       lstm2_1 [label=<<B>1</B>>];
       lstm2_2 [label=<<B>2</B>>];
       lstm2_3 [label=<<B><FONT POINT-SIZE="20">⋯</FONT></B>>];
       lstm2_512 [label=<<B>512</B>>];
   }

   subgraph cluster_lstm3 {
       label=<<B>LSTM Layer-3</B>
(512 units)>;
       style=solid;
       color="#FF6633";
       node [style=filled, color="#FF794D"];
       lstm3_1 [label=<<B>1</B>>];
       lstm3_2 [label=<<B>2</B>>];
       lstm3_3 [label=<<B><FONT POINT-SIZE="20">⋯</FONT></B>>];
       lstm3_512 [label=<<B>512</B>>];
   }

   subgraph cluster_lstm4 {
       label=<<B>LSTM Layer-4</B>
(512 units)>;
       style=solid;
       color="#FF4400";
       node [style=filled, color="#FF5C1A"];
       lstm4_1 [label=<<B>1</B>>];
       lstm4_2 [label=<<B>2</B>>];
       lstm4_3 [label=<<B><FONT POINT-SIZE="20">⋯</FONT></B>>];
       lstm4_512 [label=<<B>512</B>>];
   }

   // Dropout Layer
   subgraph cluster_dropout {
       label=<<B>Dropout Layer</B>>;
       style=dashed;
       color="#2E8B57";
       node [style=filled, color="#98FFB3"];
       dropout [label=<<B>Dropout</B>>];
   }

   // Fully Connected Layer
   subgraph cluster_fc {
       label=<<B>Fully Connected Layer</B>
(512→88)>;
       style=dashed;
       color="#1E56A0";
       node [style=filled, color="#D6E5FF"];
       fc [label=<<B>FC</B>>];
   }

   // Output Layer
   subgraph cluster_output {
       label=<<B>Output Layer</B>
(88 features)
Sigmoid Activation>;
       style=dashed;
       color="#002B5C";
       node [style=filled, color="#A3C2FF"];
       output1 [label=<<B>1</B>>];
       output2 [label=<<B>2</B>>];
       output3 [label=<<B><FONT POINT-SIZE="20">⋯</FONT></B>>];
       output88 [label=<<B>88</B>>];
   }

   // Forward Connections with thicker lines
   edge [penwidth=2.0];
   {input1 input2 input3 input88} -> {lstm1_1 lstm1_2 lstm1_3 lstm1_512} [color="#004AAD"];
   {lstm1_1 lstm1_2 lstm1_3 lstm1_512} -> {lstm2_1 lstm2_2 lstm2_3 lstm2_512} [color="#FF884D"];
   {lstm2_1 lstm2_2 lstm2_3 lstm2_512} -> {lstm3_1 lstm3_2 lstm3_3 lstm3_512} [color="#FF6633"];
   {lstm3_1 lstm3_2 lstm3_3 lstm3_512} -> {lstm4_1 lstm4_2 lstm4_3 lstm4_512} [color="#FF4400"];
   {lstm4_1 lstm4_2 lstm4_3 lstm4_512} -> dropout [color="#FF4400"];
   dropout -> fc [color="#2E8B57"];
   fc -> {output1 output2 output3 output88} [color="#1E56A0"];
}