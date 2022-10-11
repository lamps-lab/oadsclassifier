import re
#--------------------------Function for eliminating publisher url from dataset----------------------------

def eliminiate_publisher_url(features,publishers_list):
  features_without_pub_url = []
  labels_without_pub_url = []
  for i in range(0,len(features)):

    if features[i] == "\n":
      print("Empty")
    else:
      # CHECK publisher URL
            urls = re.findall(r'(https?://\S+)',features[i])
            
            temp = 0
            for j in range(0,len(urls)):
              for k in range(0,len(publishers_list)):
                if (urls[j].find(publishers_list[k]) != -1):
                  temp+=1
          
            if temp==len(urls) and len(urls)!=0:
              print("Publisher URL Found in data")
            else:
              features_without_pub_url.append(str(features[i]))
              
  return features_without_pub_url
