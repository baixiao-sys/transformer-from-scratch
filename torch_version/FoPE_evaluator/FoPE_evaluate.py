import math

import torch
import torch.nn.functional as F

from torch_version.model import DecoderOnly

from torch_version.config import d_model, n_heads, d_ff, num_blocks, rope_model_weight_path, fope_model_weight_path

from torch_version.tokenizer import CharTokenizer,BPE

from torch_version.DataVisualization.fope_rope_comparision import draw_fope_rope

seq_lens=[128,256,512,1024,2048,4096]
def evaluate_long_context(model,tokenizer,device,seq_lens=[128,256,512,1024,2048,4096]):
    model.to(device)
    model.eval()

    results ={}

    test_text="""
    Jim was very frightened when he arrived at the barber's shop. It was his first time getting a haircut! He didn't want his mum to leave him alone with the strange man.
"Don't worry," said the barber. He smiled and handed Jim a big lollipop. "I'll look after you and give you the best haircut ever."
Jim's frown disappeared and he couldn't help but be excited. The barber put a cape around him, and the chair felt like a magic carpet ride. Soon, Jim felt his hair being trimmed, and the barber even gave him some funny glasses to look through. When Jim looked in the mirror, he saw the best haircut ever! He was so happy, he forgot all about being scared.
Mum smiled when she saw Jim. She was so proud of him for sitting so still for such a long time.
"Well done, my boy!" she said, giving him a big hug. 
Jim smiled and couldn't wait to show his friends his new haircut.
<|endoftext|>

One cold winter day, Tilly was outside helping in the garden. She grabbed a rake and started cleaning up the leaves on the ground. She held the rake in front of her and carefully balanced the leaves. The ground was icy so Tilly took some small steps to steady herself.
Tilly carefully moved the rake back and forth against the ground, pushing the leaves into a pile. She smiled as the pile grew bigger. She even found some shiny objects under the leaves.
When she was finished, Tilly stood back and admired her work. The ground was so clean and tidy. Tilly was so proud of herself. She gave herself a big hug to celebrate. 
Tilly grabbed her rake, feeling warm and happy inside. She had done an amazing job with the rake and was super proud of herself.
<|endoftext|>

Once upon a time there was an adorable little rabbit. His mommy told him to rake the leaves in the garden. So he raked and raked but it was a lot of work and he got tired very quickly.
His daddy saw him struggling and decided to help him. He took the rake and together they finished raking the garden in no time. 
After they finished, his daddy said to him, "Helping each other can make chores seem like less of a chore. It's so much nicer to do them together, isn't it?". 
The adorable little rabbit nodded and smiled in agreement. Then his mommy asked him to wipe the table and the two of them wiped it together. 
In the end, the little rabbit learned a valuable lesson: when you have big tasks ahead of you, it's okay to ask for help. Working together makes chores much easier!
<|endoftext|>
 
Once upon a time there was a brother and sister who loved to play together. One day they went to the park and saw the most interesting mug. It was big, colourful and looked like it belonged to a magical creature. The brother was very tempted to steal it but the sister was obedient and wouldn't let him.
The brother said "Come on, just take it! It'll be fun!"
The sister replied firmly "No! That would be wrong, stealing is not allowed!"
Eventually, the brother listened to his sister and put the mug back. Both of them were glad that they had been obedient and followed the rules. After that, they ran off and played their favourite game. 
The End.
<|endoftext|>

Once upon a time, there was a tall fox who lived in the forest. She was very curious, and every day she liked to study the forest animals. One day, she asked a rabbit who was hopping by, "Do you know what I study every day?"
The rabbit looked up and said, "No, I don't know, what do you study?".
The fox replied, "I like to study the animals that live in the forest. I like to find out how they live and what they do."
The rabbit said, "That sounds very interesting! I wish I could study too."
The fox smiled and said, "You can. Just come with me tomorrow, and I will show you how to study the animals in the forest".
The rabbit was very happy and the next day they both went off together to study the animals in the forest. They had lots of fun, and they both learnt a lot.
<|endoftext|>
Once upon a time there was a modest lake in the middle of the forest. All the animals around were waiting to welcome a new visitor.
One day, a little 3 year old girl arrived at the lake. The animals were so excited to see a human for the first time! The squirrel welcomed her with a friendly “hi” and the deer said “hello”. The little girl was so happy to meet the animals and she replied “hi! It’s so nice to meet you!”
The little girl roamed around the lake and animals went with her. She laughed and made so many friends. All the animals were so happy that she came to visit the modest lake.
They spent a lovely day together playing and exploring the lake. In the end, they all said goodbye and the little girl said “Thank you so much for welcoming me! I had so much fun!”. And with that, she waved goodbye and continued her journey.
<|endoftext|>

One day, Jane and her daddy were at the park. Jane saw something very special - a big green tree with green powder around it. Jane walked closer to the tree and saw that the powder was made of tiny green pieces. 
"What is that Daddy?" Jane asked. 
"That's special powder, Jane," her daddy replied. "Let's gather some of it together and take it home with us."
So Jane and her daddy started gathering the powder into a bucket. They filled the bucket up with the powder and started walking home. 
When they got home, Daddy asked Jane what she wanted to do with the green powder. 
"Let's make a special cake with it," she said.
So, Jane and her daddy used the green powder to make a special cake. They cut it up and shared it between them. Jane smiled, as it was the yummiest cake she had ever tasted. 
The end.
<|endoftext|>

It was a sunny day and Jimmy was out playing in the park. As he ran around, he came across a mouse. He was excited to see it and noticed how flexible the little mouse was.
Jimmy exclaimed, “Wow, that mouse is so flexible! Maybe it wants to be my friend!”
The mouse gave him a little surprise, by hopping right up on his shoulder! Jimmy was overjoyed and thought to himself, “Now I have a new best friend!”
Jimmy and the mouse were having so much fun running around the park together, but then suddenly they heard someone coming towards them. It was Jimmy’s mom. She had come to take him home.
Jimmy shouted, “Mom, look, I made a new friend, a mouse!”
But to Jimmy’s surprise, his mom said, “Oh no, we can’t keep it. Poor thing, let it go.”
Jimmy sadly said goodbye to his new friend, and watched as the mouse ran away. He realized he would never see it again.
The end.
<|endoftext|>

Once, there were two friends who were playing in the yard. One was clumsy and the other wasn't. 
The clumsy one said, “Let’s catch a bug!”
But the other one disagreed. 
The clumsy friend said, “Why don’t you want to catch a bug?” 
The other friend said, “Bugs can be scary. Let’s play hide and seek instead.” 
The clumsy one disagreed. “No, let’s catch a bug!” 
Then, the two friends had an idea. They decided to play both games. First, they would catch a bug and then after, they would play hide and seek in the yard. 
They ran around the yard, chasing the bugs and having lots of fun. The clumsy one ran clumsily and the other one ran quicker and caught more bugs. 
The two friends laughed and played together until they were tired. They went home feeling happy and excited.
<|endoftext|>

One day, Little Bear and his Mommy went for a visit. She took him to a place he had never seen before. It was an icy place with lots of snow. As they walked around, Little Bear heard some music. It came from something his Mommy called a radio. It made Little Bear smile. 
When Little Bear asked his Mommy about it, she said it was called a radio and that it played music. 
Little Bear said, "Can I make music with it too?"
Mommy said, "Yes, Little Bear, you can make your own music too. Let's try it together."
So Little Bear and Mommy made music together with the radio. Little Bear was so happy. It was an amazing visit!
<|endoftext|>

Once upon a time, a little girl named Nora wanted to explore nature. She went to a meadow filled with white flowers, and she picked one that was the prettiest. She smelled it, and it was so sweet.
Then Nora saw a small bunny hopping in the grass. She giggled and smiled, and she softly called the bunny over to her. The bunny hopped closer and closer until it was right beside her. Nora was so happy, so she bent down and kissed the bunny on its nose.
The bunny was very happy, so it hopped off into the meadow and found it's family. Nora smiled and looked out into the meadow, taking in all the whiteness and beauty of nature. It was so gorgeous, like a picture from a fairy tale.
Nora kept playing until the sun started to go down. She thanked nature for this amazing day and kissed her hand and blew it towards the sky.  And with that, Nora went home and went to bed, dreaming of all the wonderful things nature had to offer.
<|endoftext|>

    """
    for seq_len in seq_lens:
        input_ids = tokenizer.encode(test_text)
        #转成Tensor
        input_ids = torch.tensor([input_ids[:seq_len]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits,_ = model(input_ids)
            logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            targets = input_ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits,targets,ignore_index=-1)
            ppl = math.exp(loss)

        results[seq_len] = {"loss":loss.item(),"ppl":ppl}
        #print(f"长度 {seq_len}|Loss:{loss.item():.4f}|PPL:{ppl:.2f}")
    return results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BPE()
tokenizer.load("../BPE_text.json")
vocab_size =tokenizer.get_vocab_size()

checkpoint = torch.load("../checkpoints/rope_checkpoint_last.pt")
rope_model = DecoderOnly(vocab_size,d_model, n_heads, d_ff, num_blocks)
rope_model.load_state_dict(checkpoint["model_state_dict"])
#rope_model.load_weight("../rope_model_weight.pth")
rope_model = rope_model.to(device)

checkpoint2 = torch.load("../checkpoints/fope_checkpoint_last.pt")
fope_model = DecoderOnly(vocab_size,d_model, n_heads, d_ff, num_blocks)
fope_model.load_state_dict(checkpoint2["model_state_dict"])
#fope_model.load_weight("../fope_model_weight.pth")
fope_model = fope_model.to(device)

rope_results = evaluate_long_context(rope_model, tokenizer, device)
fope_results = evaluate_long_context(fope_model, tokenizer, device)

rope_ppl = [rope_results[len]["ppl"]for len in seq_lens]
fope_ppl = [fope_results[len]["ppl"]for len in seq_lens]
draw_fope_rope(rope_ppl,fope_ppl)