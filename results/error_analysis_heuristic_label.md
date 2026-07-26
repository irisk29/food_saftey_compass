# Error Analysis — heuristic_label

201 false positives and 12 false negatives out of 1500 evaluated reviews.

Failure modes are assigned by the rule set in `analysis/error_analysis.py`. Each error may match several modes; the table counts the highest-priority one. Buckets are reproducible by construction — verify a sample per bucket by hand and report the agreement rate rather than trusting them blind.

## False positives — flagged a hazard that is not there

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `illness_mentioned_not_caused_here` | 96 | 48% | 179 |
| `unexplained_fp` | 43 | 21% | 160 |
| `negated_hazard` | 25 | 12% | 154 |
| `neutral_allergen_mention` | 23 | 11% | 231 |
| `secondhand_or_hearsay` | 5 | 2% | 230 |
| `hypothetical_or_speculative` | 4 | 2% | 92 |
| `unpleasant_not_unsafe` | 3 | 2% | 362 |
| `generic_complaint_no_hazard` | 2 | 1% | 242 |

### `illness_mentioned_not_caused_here` — 96 cases

Illness vocabulary with no causal link to this meal — 'picking up food for a sick friend', 'I was sick that week so I craved soup'. The keyword rule cannot represent causation at all, only co-occurrence, so every one of these is guaranteed to be mislabelled. A contextual model should beat the label here, which means these cases are where the model looks *wrong* while actually being right.

- I usually try and go organic whenever I can and this place isn't so bad. Wish they had more room and places to sit because sometimes it can get busy. Coffee wasn't so bad and food was decent. This is really the only place I haven't gotten sick from on the island since my husband …
- You sick sick sad little man who ran from us because you didn't know what a gramcracker is I dislike you you are almosty single handedly worse then carpinteria campground and my experience with rubbing alcohol. You have ruined the store with a giant selection of mayonnaise for me…
- Yikes! We stopped in for the lunch buffet because we were at the doing laundry next door and we were hungry. The restaurant was so dirty! The lettuce at salad bar looked old and brown, so I chose the spinach. But then I did not even want to get a fork for my salad because the ute…

### `unexplained_fp` — 43 cases

No rule matched — requires manual review.

- I like the ice cream at icesmile, the only thing that I thought sucked was the fact that they advertise giving you a free ice cream on your bday. Bring your ID they say. I was there buying more than one ice cream on the day before my bday. I asked if I could get my ice cream that…
- Family reunion was held on May 31st thru June 4th 2018 at Shephard's. Gulf front room with a great view and that's about all I could honestly say. There is really not enough light in the room especially for women when wanting to get dressed and apply makeup. There isn't a mirror …
- So I REALLY like this place, they have great coffee, soy and almond milk, and good food. (I have an allergy that prevents me from eating most of their food - but what I have eaten is great and everyone knows loves it), BUT it is a coffee shop and they take WAY too long to make a …

### `negated_hazard` — 25 cases

A hazard term inside a negation scope ('never got sick here'). Bag-of-words baselines cannot represent negation at all; a transformer can in principle, so residual errors here indicate the fine-tune did not have enough negated examples to learn it.

- Ordered food - the soup dumplings. 4 of them were fine (they tasted OK), but the other 4 were absolutely raw (took 1 bite, threw the rest away)....I hope I do not get sick from the dumplings. The other things were fine...but the raw dumplings were unacceptable.…
- This place was built onto the side the hotel I was staying in, so after driving all day I decided to have dinner with four other members of my party at Syberg's. Obviously the location was convenient. The ambiance is identical to just about any other sports bar and grill you've e…
- I haven't been here in years, and a couple of weeks ago I decided to give it another try, since I had once praised it so highly. The salad was iceberg and nothing else. That creamy ginger dressing I once loved? Too heavy and too mayo like. The sushi was fine. It was edible and I …

### `neutral_allergen_mention` — 23 cases

Allergen vocabulary used as a neutral factual note ('they have a gluten-free menu'). The model has learned allergen words predict hazards because the label was built from an allergen keyword list — this is the labelling rule leaking into the model.

- I used to give this spot 5 stars. But after my last visit. I wish I could give it ZERO. This has been a regular weekend spot for me and my man after we found out they so kindly were serving Gluten Free pancakes. (It's not on the menu but something that is known). I spread the wor…
- Upon reading the description of the 'Ziggy burger', (there was no warning of pickles) I ordered it and was disappointed. Here's why: It contained pickles, and I am allergic to white vinegar, since I was so hungry, I pulled the pickles out and ate the burger anyhow. I immediately …
- It wasn't great but it wasn't bad. The food was good. I had a ham and cheese griller so you really can't mess that up. My mom had a club and my sister had a wrap. The wrap had honey mustard even though my sister said "no honey mustard". But I don't know if I can blame the cook or…

### `secondhand_or_hearsay` — 5 cases

Hazard attributed to someone else or to other reviews. Needs source attribution, not just topic detection.

- Sound quality was pretty good and we were within a few feet of the stage for the Brandi Carlile concert (which was A+, 10 stars! but that's a separate issue from the venue). Definitely a marginally sketchy area and entry is via back alley. Everybody had to exit via literally a si…
- This place is pretty good but... if you like onions on your pizza be ready for a rude surprise- this place refuses to use onions (apparently someone there is allergic to onions... weird choice in work if that's your allergy). If you absolutely hate/ allergic to onions, this place…
- I just recently ate at Chili's at about 8:00 PM on 10/29/2017. I was greeted by a waitress who insisted on using terms of endearment towards me while calling my male companion "sir". I ordered the white spinach queso dip and we each ordered beers. The dip was amazing. The waitres…

### `hypothetical_or_speculative` — 4 cases

Hazard raised as a possibility, not an event ('I'm surprised I didn't get sick'). Requires modality/irrealis detection, which neither model is trained for.

- Maybe the pizza is good here... but I can really only speak on the soup and salad. It could be the case that I'm the idiot who decides to get italian wedding soup at a pizza joint, but it was rancid :( I tried to get through eating it, but I just couldn't do it. I was too worried…
- One star is generous. Awful, rude "service". Disgusting floors. Dirty tables. Ineffective, apathetic, rude manager. We never even got food and I'm glad because by the looks of the place, chances are you're bound to get sick. Metro Diner, just down 86th is fantastic. Go there.…
- Went to this Applebee's location with my girlfriend on 10/10/14 and it was ok. We ordered the 2 for 20 specials. I got the chicken penne and she got the steak. We both had mozzarella sticks with the order and both enjoyed it. When our food came it was different. I enjoyed my penn…

### `unpleasant_not_unsafe` — 3 cases

Describes filth or disgust without an adverse event. The boundary is a genuine definitional question — arguably these deserve flagging in a real deployment.

- This place suffers from all the typical symptoms of a poorly run business. First of all, if your pizza isn't undercooked, it's probably bland as hell. They messed up my order the first time, so I gave them a second change and the service was incredibly bad. Imagine a bunch of hig…
- I wish there were half stars, because the El dorado's buffet is definitely more of a 3.5 star. Everytime we are in Reno, this is our go to buffet . It's generally very good and consistent. However, this last visit of ours (last weekend) has somewhat changed my mind unfortunately.…
- Pretty gross. Waitress was dull and indifferent. It took her forever to bring us our food and the place was dead. Maybe cause she was chatting with people instead of checking on our food. Then the food came and it was disgusting and soggy looking. It makes me sick just thinking a…

### `generic_complaint_no_hazard` — 2 cases

An ordinary bad review with no hazard vocabulary whatsoever. If the model flags these, it is reading general negativity as danger — check whether the star gate in the label taught it that.

- As a patron with Celiac Disease, I appreciated the option to order the egg scramble. As delicious as it sounded, however, my eggs were cold when they arrived. Yuck. Our server was also not friendly or welcoming. I asked his recommendation of coffee, and he responded with a curt, …
- Unfortunately, Yelp requires placement of at least one star, in order to post a review, but, this restaurant does not rate that high. I, and my friends and family visited this restaurant this evening. We were disappointed in the food ordered, but more importantly, we were absolut…

## False negatives — missed a real hazard

| Failure mode | Count | Share | Mean words |
|---|---:|---:|---:|
| `unexplained_fn` | 6 | 50% | 184 |
| `unsafe_handling_no_illness` | 4 | 33% | 118 |
| `contamination_no_illness` | 2 | 17% | 116 |

### `unexplained_fn` — 6 cases

No rule matched — requires manual review.

- I was craving a burger, and since we'd driven past this place a few times, we decided to check it out! Pros: The chili cheese fries were REALLY good! The chili tasted and looked homemade (not the canned stuff) and was a huge portion. That could have been my entire meal by itself.…
- Let's start with my first impression ... it wasn't great. On the door walking in there are images of the credit cards they accept but when you walk to the counter there is a hand written sign that says something to the effect of we do not accept credit cards use the ATM next door…
- My wife and her friends returned to try this place again and her report is that it's worse than before. The staff is a bunch of inexperienced kids for whom this must be their first job, the food is still crappy (but not microwaved, the owner will tell you that), and the bathrooms…

### `unsafe_handling_no_illness` — 4 cases

Unsafe practice described (raw, spoiled, bare hands, hygiene) without anyone falling ill. Genuinely a hazard, but the keyword list was built from allergy/illness terms, so the label misses a share of these — the model inherits the blind spot.

- Employees need to be coached in hygiene and proper food handling techniques. Was a frequent visitor but that one gross employee made me notice other things and question whether I would come back and as of yet I have not. I messaged them previously and didn't receive any replies. …
- I've ordered pizza here twice because I am in the neighborhood. Both times have been awful. Today, the dough on my pizza was literally raw. I really want this place to suceed, but the pizza is not good.…
- We stopped in after picking my friend up at the airport. I should have read the reviews before hand as we experienced the SAME ISSUES on undercooked steaks! I should have known it was going to happen as the server asked us to cut into our steak while she waited. Both were suppose…

### `contamination_no_illness` — 2 cases

A foreign object or tampering, without illness vocabulary. Same inherited blind spot as above; this is where the heuristic's recall is weakest (88.5% on contamination).

- I had one of the top three worst restaurant experiences of my life here. One example is that my wife got a lukewarm burrito and sent it back after talking with the manager. The burrito came back partly eaten....by the chef....as the manager said she has the chef taste things when…
- There was a roach that crawled behind my shoulder on the seat. We killed it with a napkin and we were so hungry and know things are crazy with the storm coming. Once I received my food I had a hair in it. The waiter came back with a new plate after walking without saying a word a…
