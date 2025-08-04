export const events = (obj, type, callback, opt) => {

  if (obj) {

    obj.addEventListener(type, callback, opt);
  }
};

export const get_position = (element) => {

  let y_position = 0;

  while (element) {

    y_position += element.offsetTop - element.scrollTop;
    element = element.offsetParent;
  }
  
  return y_position;
};