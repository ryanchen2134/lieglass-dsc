Multimodal Dialog Deception Dataset
=====================================================
Felix Soldner, Veronica Perez-Rosas and Rada Mihalcea


Version 1.0
June 2019


CONTENTS
1. Introduction
2. Dataset Description
3. Feedback
4. Citation Information
5. Acknowledgments


======================
1. Introduction

This document describes the dataset used in the paper: Box of Lies: Multimodal Deception Detection in Dialogues. 


======================
2. Dataset Description

The dataset is derived from deceptive conversations between participants playing The Tonight Show Starring Jimmy Fallon Box of Lies game (https://www.nbc.com/the-tonight-show), in which they take turns trying to guess whether an object description provided by their opponent is deceptive or not. During each turn the participants pick a box (from among nine available boxes) that contains an object they have to describe to their opponent. The opponent must guess if the provided description is truthful or not. The participant with the best of three guesses wins the game. 

We provide multimodal annotations conducted at utterance level for 25 conversations featuring Jimmy Fallon and different guests. The conversations portray 26 unique participants, with 6 of them being males and 20 females. 

The original videos can be retrieved at https://www.youtube.com/watch?v=QhJIA8moL5s&list=PLykzf464sU99WZmHTXnkiXyk21LGN3B_X. The full video set consists of 2 hours and 24 minutes of video. The average length of a video is six minutes and contains around three rounds of the game (this varies depending on the score and on whether additional time was available for extra rounds).


2.a Annotations

The annotations are generated using a multi-level tier structure using the Elan Software(Wittenburg et al., 2006). 

We provide 25 eaf files, corresponding to the 25 recordings in our dataset. Each file represents an annotated video clip from the show “Box of Lies” of the late night talk show with Jimmy Fallon. Each file contains annotations for the Host (Jimmy Fallon) and his guest for non-verbal and verbal as well as feedback behaviors. The gesture annotations are performed using the MUMIN coding scheme (Allwood, Jens, et al. “The MUMIN multimodal coding scheme.” NorFA yearbook 2005 (2005): 129-157.). Eight gesture categories consisting of several facial displays and two categories for conversational feedback are annotated following MUMIN's guidelines. In addition, veracity was annotated on the utterance level, as being truthful or deceptive.

Gestures in each category are listed below:

General face:
	- Smile 
	- Laughter 
	- Scowl 
	- Other

Eyebrows:
- Frowning
- Raising
- Other

Eyes: 
- Exaggerated Opening 
- Closing-both 
- Closing-one 
- Closing-repeated
- Other

Gaze:
- Towards interlocutor
- Up 
- Down 
- Sideways 
- Other

Mouth Openness: 
- Open mouth
- Closed mouth

Mouth Lips: 
- Corners up 
- Corners down
- Protruded 
- Retracted

Head Movements:
- Single Nod (Down) 
- Repeated Nods (Down) 
- Move Forward 
- Move Backward 
- Single Tilt (Sideways) 
- Repeated Tilts (Sideways) 
- Side-turn 
- Shake (repeated) 
- Waggle 
- Other

Feedback annotations are listed below:

Feedback receiving:
        Acceptance
        Non-acceptance
        Continuation perception
        Continuation acceptance 


Feedback eliciting:
        Acceptance
        Non-acceptance
        Continuation perception
        Continuation acceptance

The veracity annotation was performed for each verbal statement. During the veracity coding, we assume that the behavior is always deceptive unless the verbal description indicates otherwise (i.e., accurate description of the object), as the general goal of each participant is to deceive their opponent. Thus, the veracity annotation is as follows.

Veracity:
- Truthful
- Deceptive

In addition, we provide the transcription of each speaker utterance as a separate tier using the following structure.

Verbal:
- Transcript

The following table lists each eaf file along with the caption associated with the Box of Lies conversation.

+--------------+----------------------------------------+
| Eaf filename | Box of Lies Video Caption                  |
+--------------+----------------------------------------+
| 1.BoL.An.Tr  | Box of Lies with Jennifer Lawrence         |
+--------------+----------------------------------------+
| 2.BoL.An.Tr  | Box of Lies with Channing Tatum            |
+--------------+----------------------------------------+
| 3.BoL.An.Tr  | Box of Lies with Adele                     |
+--------------+----------------------------------------+
| 4.BoL.An.Tr  | Box of Lies with Emma Stone                |
+--------------+----------------------------------------+
| 5.BoL.An.Tr  | Box of Lies with Scarlett Johansson        |
+--------------+----------------------------------------+
| 6.BoL.An.Tr  | Box of Lies with Melissa McCarthy          |
+--------------+----------------------------------------+
| 7.BoL.An.Tr  | Box of Lies with Nicole Kidman             |
+--------------+----------------------------------------+
| 8.BoL.An.Tr  | Box of Lies with Tina Fey Part 1           |
+--------------+----------------------------------------+
| 9.BoL.An.Tr  | Box of Lies with Tina Fey Part 2           |
+--------------+----------------------------------------+
| 10.BoL.An.Tr | Box of Lies with Julie Bowen               |
+--------------+----------------------------------------+
| 11.BoL.An.Tr | Box of Lies with Kate Hudson -- Part 1     |
+--------------+----------------------------------------+
| 12.BoL.An.Tr | Box of Lies with Kate Hudson -- Part 2     |
+--------------+----------------------------------------+
| 13.BoL.An.Tr | Box of Lies with Heidi Klum                |
+--------------+----------------------------------------+
| 14.BoL.An.Tr | Box of Lies with Kerry Washington          |
+--------------+----------------------------------------+
| 15.BoL.An.Tr | Box of Lies with Vince Vaughn              |
+--------------+----------------------------------------+
| 16.BoL.An.Tr | Box of Lies with Julianne Moore            |
+--------------+----------------------------------------+
| 17.BoL.An.Tr | Box of Lies with Matt Damon                |
+--------------+----------------------------------------+
| 18.BoL.An.Tr | Box of Lies with Lena Dunham               |
+--------------+----------------------------------------+
| 19.BoL.An.Tr | Box of Lies with Russell Crowe             |
+--------------+----------------------------------------+
| 20.BoL.An.Tr | Box of Lies with Emily Blunt               |
+--------------+----------------------------------------+
| 21.BoL.An.Tr | Box of Lies with Megyn Kelly               |
+--------------+----------------------------------------+
| 22.BoL.An.Tr | Box of Lies with Alec Baldwin              |
+--------------+----------------------------------------+
| 23.BoL.An.Tr | Box of Lies with Gal Gadot                 |
+--------------+----------------------------------------+
| 24.BoL.An.Tr | Box of Lies with Halle Berry               |
+--------------+----------------------------------------+
| 25.BoL.An.Tr | Box of Lies with Margot Robbie             |
+--------------+----------------------------------------+   |


2.b Usage

The eaf files can be processed  as follows:

* Using ELAN (Wittenburg et al., 2006), which can be freely downloaded at:
https://tla.mpi.nl/tools/tla-tools/elan/download/        
Using this option, users can retrieve and navigate the multi-tiered structure of the annotations. Elan provides export options so the different annotation layers can be saved in plain text format.

* Using “Pympi” (Lubbers and Torreira, 2013), a python package that enables reading and modifying eaf files.


3. Feedback
For questions or inquiries regarding this dataset, you can contact Felix Soldner, Veronica Perez-Rosas and Rada Mihalcea
felix.soldner@ucl.ac.uk 
vrncapr@umich.edu
mihalcea@umich.edu


4. Citation Information

Bibtex:
@article{SoldnerDeceptionDialog,
author = {Felix Soldner, Ver\’{o}nica P\'{e}rez-Rosas, Rada Mihalcea},
title = {Box of Lies: Multimodal Deception Detection in Dialogues},
journal = {2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
year = {2019}
}

Text: 
Felix Soldner, Veronica Perez-Rosas, and Rada Mihalcea. 2019. Box of Lies: Multimodal Deception Detection in Dialogues in Proceedings of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics. Minneapolis, USA. 


5. Acknowledgments

This material is based in part upon work supported by the Michigan Institute for Data Science, by the National Science Foundation (grant #1815291), and by the John Templeton Foundation (grant #61156). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author and do not necessarily reflect the views of the Michigan Institute for Data Science, the National Science Foundation, or the John Templeton Foundation.


#References

Jens Allwood, Loredana Cerrato, Laila Dybkjaer, Kristiina Jokinen, Costanza Navarretta, and Patrizia Paggio. 2005. The MUMIN multimodal coding scheme. NorFA yearbook, 2005:129–157.

Mart Lubbers and Francisco Torreira. 2013. Pympiling: A Python module for processing ELANs EAF and Praats TextGrid annotation files

Peter Wittenburg, Hennie Brugman, Albert Russel, Alex Klassmann, and Han Sloetjes. 2006. ELAN: A Professional Framework for Multimodality Research. page 4.