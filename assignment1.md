# music-genre-classification

As I am very much into music and recently started learning dj-ing I will choose music genre classification as my topic. I will do a “bring your own data” project as it is very interesting for me to see this classification done on my own music.

My motivation for choosing this topic is that I actually find it often quite difficult to assign a genre to a song. Often times there are even multiple genres fitting. And especially for my music, which consists mainly of electronic dance music the subgenres “borrow” much more from each other or have many similarities. So the genres can be considered more of a spectrum, with multiple labels possibly fitting.

The dataset I am going to use will be created from the music I listen to myself. The problem will be getting accurate genre labels for these songs. Manually labeling the songs would not be accurate enough so I have to rely on some database. I found this one [rateyourmusic.com](https://rateyourmusic.com) it does not have all of the songs I checked, but it often provides multiple more granular genre labels. For example the song “We Do What We Want” by Alan Fitzpatrick lists the genres “Peak Time Techno, Acid Techno, Breakbeat” rather than just “Techno”. This would allow me to train for multiple labels for songs.

The dataset will consist of a spectrogram of a segment of the song (possibly multiple segments per song) like in \[1\] and \[2\] and a set of labels obtained from rateyourmusic.com.

For creating the dataset I will extend [an existing tool I wrote](https://github.com/GeorgSchenzel/spotdj) (I just made it public so there is no documentation yet; It is also not very stable at the moment), which takes spotify playlists and downloads the individual songs from youtube. I will extend it by also searching for the song on rateyourmusic.com and fetching the genre labels. One difficulty is that there is no API as of now, so I will have to scrape the values myself.

The training will be done either on VGG-16 as in \[2\] or on the network proposed in \[1\]. I can imagine that ResNet will also lead to similar results so it is also worth checking out.

## Time estimates
30h dataset collection  
5h designing and building an appropriate network  
10h training and fine-tuning that network  
10h building an application to present the results  
10h writing the final report; preparing the presentation of your work

# References

\[1\] Elbir, A., & Aydin, N. (2020). Music genre classification and music recommendation by using deep learning. _Electronics Letters_, _56_(12), 627-629.

\[2\] Bahuleyan, H. (2018). Music genre classification using machine learning techniques. _arXiv preprint arXiv:1804.01149_.
