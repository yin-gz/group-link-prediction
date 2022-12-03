import pandas as pd
import json
from collections import defaultdict as ddict

# load file
f = open("../weeplace_checkins.csv", encoding="utf-8")
data = pd.read_csv(f,dtype = 'object')
data['datetime'] = data['datetime'].apply(lambda x: x.split('T')[0])
item_list = data.to_dict('list')['placeid']
cate_list = data.to_dict('list')['category']

# count item frequency
item_count = ddict(int)#{'item': count}
for item in item_list:
    item_count[item] += 1

# map an item to its cate {'item': 'cate'}
# (filter out items which have no cates or frequency<50)
item2cate = {}
for i in range(len(item_list)):
    if pd.isna(item_list[i]) or pd.isna(cate_list[i]) or item_count[item_list[i]] < 50:
        pass
    else:
        if item_list[i] not in item2cate:
            item2cate[item_list[i]] = cate_list[i]
print('item number',len(item2cate))

# map a user to his friend circle
user2friend = {} #{'user': ['','',...]}
f = open("../weeplace_friends.csv", encoding="utf-8")
friends_data = pd.read_csv(f,dtype = 'object')
for user,g in friends_data.groupby(['userid1']):
    user2friend[user] = list(g['userid2'])
    user2friend[user].append(user)


# step1: group the users who visit the same place at the same time to several sets
# step2: split each user set to groups according to their friend set
# step3: add/update the group info and group-to-item info
data = data.loc[:,['userid','placeid','datetime']] #(user, time, place)
group_info = {} #{[g_member1, g_member2, ...]:group_id}
group_item = ddict(list) #{group_id:[item1,item2,...]}
for place_date, g in data.groupby(['placeid','datetime']):
    if len(place_date[0]) > 2 and (place_date[0] in item2cate): #filter
        item = place_date[0]
        time = place_date[1]
        group_id_set = set()
        v_people = set(g['userid'])
        group_list = []
        people_circle_list = []

        # split each user set to groups
        for user in v_people:
            try:
                user_friend_set = set(user2friend[user])
                flag = True
                #merge two circles if they have intersection
                for i in range(len(people_circle_list)):
                    if len(user_friend_set & people_circle_list[i])>1:
                        people_circle_list[i]= people_circle_list[i] | user_friend_set
                        flag = False
                        break
                if flag:
                    people_circle_list.append(user_friend_set)
            except:
                print(f'Error, user {user} not in weeplace_friends')

        # check if (v_people ∩ each friend circle) > 2, if yes, add/update the group info
        for people_circle in people_circle_list:
            if len(people_circle & v_people)>1:
                group_list.append(list(people_circle & v_people))
        for group in group_list:
            group.sort()
            group = tuple(group)
            if group not in group_info:
                id = len(group_info)
                group_info[group] = id
            else:
                id = group_info[group]
            group_id_set.add(id)
        for group_id in group_id_set:
            group_item[group_id].append(item)


#gengerate group info and group to item files
group_info = {v:k for k,v in group_info.items()}
f1 = open('group_info.json','w',encoding='utf-8')
f1.writelines(json.dumps(group_info))
f2 = open('group_item.json','w',encoding='utf-8')
f2.writelines(json.dumps(group_item))

#gengerate item to cate files
f2 = open('item_cate.json','w',encoding='utf-8')
f2.writelines(json.dumps(item2cate))

#gengerate user to item file
#filer out items if user visit them as a group
user2group = ddict(list)
for group,user_list in group_info.items():
    for user in user_list:
        user2group[user].append(group)
user_ingroup_item = ddict(list)
for user,group_list in user2group.items():
    for group in group_list:
        user_ingroup_item[user].extend(group_item[group])
#filter out users without groups
people_in_group = set()
for k,v in group_info.items():
    people_in_group = people_in_group | set(v)
user_item = ddict(list)
for userid,g in data.groupby(['userid']):
    if userid in people_in_group:
        item_set = set(g['placeid'])
        for item in item_set:
            if pd.isna(item) or (item not in item2cate):
                pass
            elif item in user_ingroup_item[user]:
                pass
            else:
                user_item[str(userid)].append(item)
f4 = open('user_item.json','w',encoding='utf-8')
f4.writelines(json.dumps(user_item))