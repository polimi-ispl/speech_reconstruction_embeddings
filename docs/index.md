
<nav id="menu">
	<h2>Menu</h2>
	<ul>
		<li><a href="index.md">Home</a></li>
	</ul>
	<h2>Menu</h2>
	<ul>
		<li><a href="vggish_results.md">Home</a></li>
	</ul>
</nav>

<details>
<summary>Menu</summary>
<ul><li>[Home](index.md)</li>
<li>[Vggish results](vggish_results.md)</li></ul>
<li>[SmallEnc results](smallenc_results.md)</li></ul> 
</details>


## Welcome to GitHub Pages

The complete understanding of the decision-makingprocess  of  Convolutional  Neural  Networks  (CNNs)  is  far  frombeing fully reached. Many researchers proposed techniques to in-terpret what a network actually “learns” from data. Neverthelessmany  questions  still  remain  unanswered.  In  this  work  we  studyone  aspect  of  this  problem  by  reconstructing  speech  from  theintermediate  embeddings  computed  by  a  CNNs.  Specifically,  weconsider  a  pre-trained  network  that  acts  as  a  feature  extractorfrom  speech  audio.  We  investigate  the  possibility  of  invertingthese  features,  reconstructing  the  input  signals  in  a  black-boxscenario,  and  quantitatively  measure  the  reconstruction  qualityby measuring the word-error-rate of an off-the-shelf ASR model.Experiments  performed  using  two  different  CNN  architecturestrained for six different classification tasks, show that it is possibleto   reconstruct   time-domain   speech   signals   that   preserve   thesemantic content, whenever the embeddings are extracted beforethe  fully  connected  layers.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and 
```
![Image](images/home/pipeline_1.png)

<audio controls>
<source src="audio/LJ049-0209.wav" type="audio/mpeg">
Your browser does not support the audio element.
</audio>


For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/polimi-ispl/speech_reconstruction_embeddings/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
